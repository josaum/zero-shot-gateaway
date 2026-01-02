#!/usr/bin/env python3
"""
Zero-copy shared memory consumer for the Physics Engine.

Usage:
    python consumer.py --follow        # Tail mode (like `tail -f`)
    python consumer.py --batch 1000    # Batch read mode
    python consumer.py --pytorch       # Return PyTorch tensors
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

MAGIC = 0x5048595349435300  # "PHYSICS\0"
HEADER_SIZE = 64
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
    
    Uses SeqLock protocol to detect torn reads.
    """

    def __init__(self, shm_path: str = SHM_PATH):
        self.shm_path = Path(shm_path)
        self._mm: Optional[mmap.mmap] = None
        self._array: Optional[np.ndarray] = None
        self.header: Optional[FrameHeader] = None
        self._last_read_head: int = 0

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
        self._mm = mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ)
        fd.close()

        # Validate header
        self.header = self._read_header()
        if self.header.magic != MAGIC:
            raise ValueError(f"Invalid magic: {self.header.magic:#x}, expected {MAGIC:#x}")
        if self.header.frame_size != FRAME_SIZE:
            raise ValueError(f"Frame size mismatch: {self.header.frame_size} != {FRAME_SIZE}")

        # Create numpy view of frame data (zero-copy!)
        # Offset by HEADER_SIZE
        # We need to map the whole file first (done in _mm), then create a numpy wrapper.
        # np.frombuffer behaves differently with mmap, so we'll be careful.
        
        # Calculate max size to map
        data_len = self.header.capacity * FRAME_SIZE
        
        # We access via slicing `self._mm` which is efficient
        # To get a zero-copy numpy array over the whole data section:
        # Note: We need to be careful about bounds.
        self._buffer_view = memoryview(self._mm)[HEADER_SIZE:]
        # We can also create a typed view if needed, but we do per-frame reads to be safe with SeqLock

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
        # Note: tail is optional in Rust struct, but we read 8 bytes.
        # Padding ensures alignment.
        return FrameHeader(
            magic=magic,
            version=version,
            frame_size=frame_size,
            capacity=capacity,
            head=head,
            tail=tail,
        )

    def _read_frame_at(self, index: int, max_retries: int = 3) -> Optional[ParticleFrame]:
        """
        Read a single frame using SeqLock protocol.
        
        Returns None if frame is being written (retry exceeded).
        """
        offset = HEADER_SIZE + (index * FRAME_SIZE)  # Absolute offset in file
        
        # We read directly from mmap to get latest data
        
        for _ in range(max_retries):
            # 1. Read sequence (must be even = not being written)
            # Read 8 bytes at offset
            seq_bytes = self._mm[offset : offset + 8]
            seq1 = struct.unpack("<Q", seq_bytes)[0]
            
            if seq1 & 1:  # Odd = write in progress
                time.sleep(0.000001)  # 1μs backoff
                continue

            # 2. Read payload
            # We copy the frame bytes to avoid it changing while we parse
            # This is "Zero-Copy" in the sense that we don't serialize/deserialize JSON,
            # but for SeqLock safety in Python we might copy the 4k block once.
            # To be truly zero-copy we would read fields from the view, but that risks torn reads 
            # if we are too slow. 
            # Given Python's speed, copying 4KB is safer and still microsecond-scale.
            frame_bytes = self._mm[offset : offset + FRAME_SIZE]

            # Parse fields
            # Layout:
            # 0-8: sequence
            # 8-16: frame_id
            # 16-4112: semantic (1024 floats)
            # 4112-...: kinetics
            
            frame_id = struct.unpack("<Q", frame_bytes[8:16])[0]
            
            # Semantic: 1024 floats (4096 bytes) at offset 16
            semantic = np.frombuffer(frame_bytes[16:4112], dtype=np.float32).copy()
            
            # Kinetics
            # 4112: delta_time (f32)
            # 4116: duration_ms (f32)
            # 4120: velocity (f32)
            # 4124: interrupt (f32)
            # 4128: timestamp_us (i64)
            # 4136: conversation_hash (u64)
            kinetics_floats = struct.unpack("<ffff", frame_bytes[4112:4128])
            delta_time, duration_ms, velocity, interrupt = kinetics_floats
            
            timestamp_us = struct.unpack("<q", frame_bytes[4128:4136])[0]
            conversation_hash = struct.unpack("<Q", frame_bytes[4136:4144])[0]

            # Positional
            # 4144: workflow_node (i32)
            # 4148: case_status (i32)
            # 4152: lead_status (i32)
            # 4156: fill_rate (f32)
            # 4160: actor_type (i32)
            # 4164: intent_id (i32)
            # 4168: sentiment (f32)
            # 4172: confidence (f32)
            position_ints_1 = struct.unpack("<iii", frame_bytes[4144:4156])
            workflow_node, case_status, lead_status = position_ints_1
            
            fill_rate = struct.unpack("<f", frame_bytes[4156:4160])[0]
            
            position_ints_2 = struct.unpack("<ii", frame_bytes[4160:4168])
            actor_type, intent_id = position_ints_2
            
            position_floats_2 = struct.unpack("<ff", frame_bytes[4168:4176])
            sentiment, confidence = position_floats_2

            # Spin: 4 floats at 4176
            spin = np.frombuffer(frame_bytes[4176:4192], dtype=np.float32).copy()

            # 3. Re-read sequence (must match)
            # We must read from the ORIGINAL mmap location, not our copy
            seq_bytes_2 = self._mm[offset : offset + 8]
            seq2 = struct.unpack("<Q", seq_bytes_2)[0]
            
            if seq1 != seq2:
                # Torn read
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

        return None  # Retry exhausted

    def read_latest(self) -> Optional[ParticleFrame]:
        """Read the most recent frame."""
        self.header = self._read_header()
        # Head points to NEXT write position usually, or current? 
        # In Rust: cursor starts 0. WRITE happens at cursor. Then cursor++.
        # So latest VALID frame is cursor - 1 (modulo size)
        latest_idx = (self.header.head - 1) % self.header.capacity
        if latest_idx < 0:
             latest_idx += self.header.capacity
        return self._read_frame_at(latest_idx)

    def read_batch(self, n: int) -> List[ParticleFrame]:
        """Read the last N frames (newest first)."""
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
        """
        Yield new frames as they arrive (like `tail -f`).
        """
        last_frame_id = -1
        
        # Initial catch-up: read latest frame to establish baseline
        latest = self.read_latest()
        if latest:
            last_frame_id = latest.frame_id - 1 # Start from current

        print(f"Listening for frames > {last_frame_id}")

        while True:
            self.header = self._read_header()
            current_head = self.header.head
            
            # Check frame at (current_head - 1)
            frame = self.read_latest()
            
            if frame and frame.frame_id > last_frame_id:
                # We found a newer frame
                # If we missed multiple, we might want to iterate from last_frame_id+1 to current
                # But for simple follow, just yielding latest valid is okay, 
                # or we can try to walk forward.
                # Let's just yield the one we found.
                last_frame_id = frame.frame_id
                yield frame
            else:
                time.sleep(poll_interval)


class PyTorchConsumer(PhysicsConsumer):
    """
    Consumer that returns PyTorch tensors directly.
    Requires: pip install torch
    """

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

    parser = argparse.ArgumentParser(description="Physics Engine Consumer")
    parser.add_argument("--follow", action="store_true", help="Tail mode")
    parser.add_argument("--batch", type=int, default=0, help="Read N frames")
    parser.add_argument("--pytorch", action="store_true", help="Use PyTorch")
    args = parser.parse_args()

    try:
        ConsumerClass = PyTorchConsumer if args.pytorch else PhysicsConsumer
        
        print(f"Connecting to shared memory at {SHM_PATH}...")
        with ConsumerClass() as consumer:
            if args.follow:
                print("Following new frames (Ctrl+C to stop)...")
                for frame in consumer.follow():
                    print(f"Frame {frame.frame_id}: conv={frame.conversation_hash:#x}, "
                          f"sentiment={frame.sentiment:.2f}, velocity={frame.velocity:.1f}")
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
