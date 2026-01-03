#!/usr/bin/env python3
"""
Zero-copy shared memory consumer for the Physics Engine.

Usage:
    python scripts/consumer.py --follow        # Tail mode (like `tail -f`)
    python scripts/consumer.py --batch 1000    # Batch read mode
    python scripts/consumer.py --pytorch       # Return PyTorch tensors
"""

import mmap
import struct
import time
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple, List

import numpy as np

# Adjust SHM_PATH for macOS support (matching Rust implementation)
if platform.system() == "Darwin":
    SHM_PATH = "/tmp/cs_physics"
else:
    SHM_PATH = "/dev/shm/cs_physics"

MAGIC = 0xCAFEBABE  # Updated V2 Magic
HEADER_SIZE = 64
# V2 Layout:
# [ Header (64b) ] [ Padding -> 4KB ] [ Control Plane (1MB) ] [ Ring Buffer ]
CONTROL_OFFSET = 4096
CONTROL_SIZE = 1024 * 1024  # 1MB
RING_OFFSET = CONTROL_OFFSET + CONTROL_SIZE
FRAME_SIZE = 4224

@dataclass
class FrameHeader:
    """Ring buffer header structure."""
    magic: int
    version: int
    frame_size: int
    capacity: int
    head: int
    tail: int

@dataclass
class Attractor:
    position: np.ndarray  # (1024,)
    mass: float
    label: bytes # 32 bytes

@dataclass
class ParticleFrame:
    """Single frame from the ring buffer."""
    sequence: int
    frame_id: int
    semantic: np.ndarray  # (1024,) float32
    delta_time: float
    duration_ms: float
    velocity: float
    interrupt: float
    timestamp_us: int
    conversation_hash: int
    workflow_node: int
    case_status: int
    lead_status: int
    fill_rate: float
    actor_type: int
    intent_id: int
    sentiment: float
    confidence: float
    spin: np.ndarray  # (4,) float32


class PhysicsConsumer:
    """
    Zero-copy consumer for the Physics Engine shared memory buffer.
    Also acts as the 'Controller' for the Gravitational Learning Control Plane.
    
    Uses SeqLock protocol to detect torn reads.
    """

    def __init__(self, shm_path: str = SHM_PATH):
        self.shm_path = Path(shm_path)
        self._mm: Optional[mmap.mmap] = None
        self.header: Optional[FrameHeader] = None

    def __enter__(self) -> "PhysicsConsumer":
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def open(self):
        """Open and memory-map the shared memory file."""
        if not self.shm_path.exists():
            raise FileNotFoundError(f"Shared memory not found: {self.shm_path}. Make sure the Gateway is running.")

        fd = open(self.shm_path, "r+b")
        self._mm = mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_WRITE) # Need WRITE for Control Plane
        fd.close()

        # Validate header
        self.header = self._read_header()
        if self.header.magic != MAGIC:
            raise ValueError(f"Invalid magic: {self.header.magic:#x}, expected {MAGIC:#x}")
        if self.header.frame_size != FRAME_SIZE:
            raise ValueError(f"Frame size mismatch: {self.header.frame_size} != {FRAME_SIZE}")

    def close(self):
        """Close the memory mapping."""
        if self._mm:
            self._mm.close()
            self._mm = None

    def _read_header(self) -> FrameHeader:
        """Read the ring buffer header."""
        header_bytes = self._mm[:HEADER_SIZE]
        magic, version, frame_size, capacity, head, tail = struct.unpack(
            "<QIIQqq", header_bytes[:40]
        )
        return FrameHeader(magic, version, frame_size, capacity, head, tail)

    # --- Control Plane Methods (Gravitational Learning) ---

    def set_gravity(self, gravity: float, friction: float):
        """Update global physics parameters."""
        # Offset 4096: global_gravity (atomic u32 from f32)
        # Offset 4100: friction
        
        # Convert float to u32 bits equivalent
        g_bytes = struct.pack("<f", gravity)
        f_bytes = struct.pack("<f", friction)
        
        # Atomic store simulation (Python GIL makes this reasonably safe against partial writes, 
        # but Rust uses atomics. Ideally we write 4 bytes fully).
        self._mm[CONTROL_OFFSET:CONTROL_OFFSET+4] = g_bytes
        self._mm[CONTROL_OFFSET+4:CONTROL_OFFSET+8] = f_bytes

    def set_attractor(self, index: int, vector: np.ndarray, mass: float, label: str):
        """
        Write an attractor to the Control Plane.
        index: 0-63
        vector: (1024,) float32 array
        mass: float
        label: string (max 32 chars)
        """
        if index < 0 or index >= 64:
            raise ValueError("Attractor index must be 0-63")
        
        # Attractor Size = 4192
        # Start of Attractors Array = CONTROL_OFFSET + 16 bytes (padding)
        base_addr = CONTROL_OFFSET + 16 + (index * 4192)
        
        # Struct:
        # position: 1024 f32 (4096 bytes)
        # mass: f32 (4 bytes)
        # label: 32 bytes
        # padding: 60 bytes
        
        # Write position
        if vector.shape != (1024,):
            raise ValueError("Vector must be shape (1024,)")
            
        vector_bytes = vector.astype(np.float32).tobytes()
        self._mm[base_addr : base_addr+4096] = vector_bytes
        
        # Write mass
        self._mm[base_addr+4096 : base_addr+4100] = struct.pack("<f", mass)
        
        # Write label
        label_bytes = label.encode('ascii', errors='ignore')[:31] # Leave room for null? Rust [u8;32] just bytes
        padded_label = label_bytes.ljust(32, b'\0')
        self._mm[base_addr+4100 : base_addr+4132] = padded_label

    def update_attractor_count(self, count: int):
        """Update the number of active attractors."""
        count = min(max(0, count), 64)
        c_bytes = struct.pack("<I", count) # u32
        self._mm[CONTROL_OFFSET+8 : CONTROL_OFFSET+12] = c_bytes

    # --- Frame Reading Methods ---

    def _read_frame_at(self, index: int, max_retries: int = 3) -> Optional[ParticleFrame]:
        # Offset calculation changed for V2
        offset = RING_OFFSET + (index * FRAME_SIZE)
        
        for _ in range(max_retries):
            # 1. Read sequence
            seq_bytes = self._mm[offset : offset + 8]
            seq1 = struct.unpack("<Q", seq_bytes)[0]
            
            if seq1 & 1:  # Odd = write in progress
                time.sleep(0.000001)
                continue

            # 2. Read payload (4224 bytes)
            frame_bytes = self._mm[offset : offset + FRAME_SIZE]

            frame_id = struct.unpack("<Q", frame_bytes[8:16])[0]
            semantic = np.frombuffer(frame_bytes[16:4112], dtype=np.float32).copy()
            
            kinetics_floats = struct.unpack("<ffff", frame_bytes[4112:4128])
            delta_time, duration_ms, velocity, interrupt = kinetics_floats
            
            timestamp_us = struct.unpack("<q", frame_bytes[4128:4136])[0]
            conversation_hash = struct.unpack("<Q", frame_bytes[4136:4144])[0]

            position_ints_1 = struct.unpack("<iii", frame_bytes[4144:4156])
            workflow_node, case_status, lead_status = position_ints_1
            
            fill_rate = struct.unpack("<f", frame_bytes[4156:4160])[0]
            
            position_ints_2 = struct.unpack("<ii", frame_bytes[4160:4168])
            actor_type, intent_id = position_ints_2
            
            position_floats_2 = struct.unpack("<ff", frame_bytes[4168:4176])
            sentiment, confidence = position_floats_2

            spin = np.frombuffer(frame_bytes[4176:4192], dtype=np.float32).copy()

            # 3. Validation
            seq_bytes_2 = self._mm[offset : offset + 8]
            seq2 = struct.unpack("<Q", seq_bytes_2)[0]
            
            if seq1 != seq2:
                time.sleep(0.000001)
                continue

            return ParticleFrame(
                sequence=seq1,
                frame_id=frame_id,
                semantic=semantic,
                delta_time=delta_time,
                duration_ms=duration_ms,
                velocity=velocity,
                interrupt=interrupt,
                timestamp_us=timestamp_us,
                conversation_hash=conversation_hash,
                workflow_node=workflow_node,
                case_status=case_status,
                lead_status=lead_status,
                fill_rate=fill_rate,
                actor_type=actor_type,
                intent_id=intent_id,
                sentiment=sentiment,
                confidence=confidence,
                spin=spin,
            )

        return None

    def read_latest(self) -> Optional[ParticleFrame]:
        self.header = self._read_header()
        latest_idx = (self.header.head - 1) % self.header.capacity
        if latest_idx < 0:
             latest_idx += self.header.capacity
        return self._read_frame_at(latest_idx)

    def read_batch(self, n: int) -> List[ParticleFrame]:
        self.header = self._read_header()
        frames = []
        head = self.header.head
        capacity = self.header.capacity

        for i in range(n):
            idx = (head - 1 - i) % capacity
            if idx < 0:
                idx += capacity
            frame = self._read_frame_at(idx)
            if frame:
                frames.append(frame)

        return frames

    def follow(self, poll_interval: float = 0.001) -> Iterator[ParticleFrame]:
        last_frame_id = -1
        latest = self.read_latest()
        if latest:
            last_frame_id = latest.frame_id - 1

        print(f"Listening for frames > {last_frame_id} (V2 Protocol)...")

        while True:
            self.header = self._read_header()
            frame = self.read_latest()
            
            if frame and frame.frame_id > last_frame_id:
                last_frame_id = frame.frame_id
                yield frame
            else:
                time.sleep(poll_interval)


class PyTorchConsumer(PhysicsConsumer):
    def __init__(self, shm_path: str = SHM_PATH, device: str = "cpu"):
        super().__init__(shm_path)
        import torch
        self.torch = torch
        self.device = device
        print(f"PyTorch Consumer initialized on {device}")

    def read_batch_tensors(self, n: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
        frames = self.read_batch(n)
        if not frames:
            return (
                self.torch.empty(0, 1024, device=self.device),
                self.torch.empty(0, 14, device=self.device)
            )
        
        semantics = np.stack([f.semantic for f in frames])
        kinetics = np.array([
            [
                f.delta_time, f.duration_ms, f.velocity, f.interrupt,
                f.timestamp_us / 1e6,
                float(f.workflow_node), float(f.case_status), float(f.lead_status),
                f.fill_rate, float(f.actor_type), float(f.intent_id),
                f.sentiment, f.confidence,
                *f.spin
            ]
            for f in frames
        ], dtype=np.float32)

        return (
            self.torch.from_numpy(semantics).to(self.device),
            self.torch.from_numpy(kinetics).to(self.device),
        )


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Physics Engine Consumer V2")
    parser.add_argument("--follow", action="store_true", help="Tail mode")
    parser.add_argument("--batch", type=int, default=0, help="Read N frames")
    parser.add_argument("--pytorch", action="store_true", help="Use PyTorch")
    parser.add_argument("--learn", action="store_true", help="Run Learning Loop (K-Means)")
    
    args = parser.parse_args()

    try:
        ConsumerClass = PyTorchConsumer if args.pytorch else PhysicsConsumer
        
        print(f"Connecting to shared memory at {SHM_PATH}...")
        with ConsumerClass() as consumer:
            if args.learn:
                print("🧠 Gravitational Learning Loop Active...")
                from sklearn.cluster import MiniBatchKMeans
                import time
                
                # Set initial physics
                consumer.set_gravity(2.0, 0.1)
                print("Set Global Gravity=2.0, Friction=0.1")
                
                while True:
                    print("Reading recent history...")
                    frames = consumer.read_batch(500)
                    if len(frames) > 50:
                        data = np.stack([f.semantic for f in frames])
                        
                        # Find clusters
                        kmeans = MiniBatchKMeans(n_clusters=5, random_state=0, batch_size=256)
                        kmeans.fit(data)
                        
                        centers = kmeans.cluster_centers_
                        print(f"Found {len(centers)} clusters via K-Means.")
                        
                        # Update Attractors
                        consumer.update_attractor_count(len(centers))
                        for i, center in enumerate(centers):
                            # Name it basically
                            name = f"Cluster_{i}"
                            # Calculate 'mass' based on cluster size (simplification)
                            mass = 10.0 
                            consumer.set_attractor(i, center, mass, name)
                            print(f"  -> Set Attractor {i}: {name} (Mass {mass})")
                        
                        print("Updated Physics Control Plane. Sleeping...")
                    else:
                        print("Not enough data yet...")
                        
                    time.sleep(5.0)

            elif args.follow:
                print("Following new frames (Ctrl+C to stop)...")
                for frame in consumer.follow():
                    print(f"Frame {frame.frame_id}: vel={frame.velocity:.2f} (Hash {frame.conversation_hash:#x})")
            elif args.batch > 0:
                frames = consumer.read_batch(args.batch)
                print(f"Read {len(frames)} frames")
                for f in frames[:5]:
                    print(f"  {f.frame_id}: semantic_sample={f.semantic[:3]}...")
            else:
                frame = consumer.read_latest()
                if frame:
                    print(f"Latest frame {frame.frame_id}: {frame.conversation_hash}")
                else:
                    print("No frames available")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run the Gateway (cargo run) first to initialize shared memory.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
