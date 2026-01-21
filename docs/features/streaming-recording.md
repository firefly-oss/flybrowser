# Streaming & Recording Features

FlyBrowser provides professional-grade video recording and live streaming capabilities for browser sessions, supporting all deployment modes (embedded, standalone, cluster).

## Overview

- **Recording**: Capture browser sessions as video files with screenshots
- **Live Streaming**: Stream browser sessions in real-time using HLS, DASH, or RTMP
- **Quality Profiles**: Optimized presets for bandwidth and quality trade-offs
- **Video Codecs**: H.264, H.265 (HEVC), VP9, VP8, AV1 support
- **Hardware Acceleration**: Automatic detection and use of NVENC, VideoToolbox, QSV
- **Distributed Storage**: S3, MinIO, NFS, or local filesystem storage
- **Low Latency**: Sub-3-second latency with HLS, <1s with DASH

## Recording

### Basic Recording

Record a browser session to a video file:

```python path=null start=null
from flybrowser import FlyBrowser

async with FlyBrowser(llm_provider="openai", api_key="sk-...") as browser:
    # Start recording
    await browser.start_recording()
    
    # Perform actions
    await browser.goto("https://example.com")
    await browser.act("click the login button")
    
    # Stop and save recording
    recording = await browser.stop_recording()
    print(f"Recording saved: {recording['recording_id']}")
    print(f"Duration: {recording['duration_seconds']}s")
    print(f"Frames: {recording['frame_count']}")
```

### Recording Configuration

Configure recording quality and format:

```python path=null start=null
# Server mode with advanced options
async with FlyBrowser(endpoint="http://localhost:8000") as browser:
    await browser.start_recording(
        codec="h265",           # h264, h265, vp9
        quality="high",         # low_bandwidth, medium, high
        enable_hw_accel=True,   # Use hardware acceleration
        fps=30,                 # Frames per second
    )
```

### Listing Recordings

List available recordings:

```python path=null start=null
# List all recordings
recordings = await browser.list_recordings()
for rec in recordings:
    print(f"{rec['recording_id']}: {rec['duration_seconds']}s - {rec['file_size_mb']}MB")

# Filter by session
session_recordings = await browser.list_recordings(session_id="sess_abc123")
```

### Downloading Recordings

Download recorded videos:

```python path=null start=null
# Download to local file
info = await browser.download_recording(
    recording_id="rec_xyz789",
    output_path="./my_recording.mp4"
)
print(f"Downloaded to: {info['file_path']}")
```

## Live Streaming

### Starting a Stream

Stream browser sessions in real-time:

```python path=null start=null
async with FlyBrowser() as browser:
    await browser.goto("https://example.com")
    
    # Start HLS stream
    stream = await browser.start_stream(
        protocol="hls",      # hls, dash, rtmp
        quality="medium",    # low_bandwidth, medium, high
        codec="h264"         # h264, h265, vp9
    )
    
    print(f"Stream URL: {stream['stream_url']}")
    print(f"View at: {stream['hls_url']}")
    
    # Perform actions while streaming
    await browser.act("scroll down slowly")
    
    # Stop stream
    result = await browser.stop_stream()
    print(f"Stream stopped: {result.get('success')}")
    
    # Access final stats if available
    if result.get('info'):
        final_metrics = result['info'].get('metrics', {})
        print(f"Total frames: {final_metrics.get('frames_sent', 0)}")
        print(f"Total bytes: {final_metrics.get('bytes_sent', 0)}")
```

### Embedded Mode Streaming

Embedded mode automatically starts a local HTTP server for streaming:

```python path=null start=null
# Embedded mode (no endpoint)
async with FlyBrowser(llm_provider="openai", api_key="sk-...") as browser:
    stream = await browser.start_stream(protocol="hls", quality="medium")
    
    # Stream served on localhost with random available port
    print(f"Local stream: {stream['stream_url']}")
    # Example: http://localhost:54321/stream_abc123/playlist.m3u8
    
    # Use with video player like VLC or web player
    # vlc {stream['hls_url']}
```

### Stream Status & Metrics

Monitor stream health and performance:

```python path=null start=null
status = await browser.get_stream_status()

# Status has nested structure
print(f"Active: {status.get('active')}")

# Access nested stream data
if status.get('active'):
    stream_data = status.get('status', {})
    print(f"Stream ID: {stream_data.get('stream_id')}")
    print(f"State: {stream_data.get('state')}")  # active, paused, stopped
    print(f"Health: {stream_data.get('health')}")  # healthy, degraded, unhealthy
    print(f"Uptime: {stream_data.get('uptime_seconds')}s")
    
    # Access metrics
    metrics = stream_data.get('metrics', {})
    print(f"FPS: {metrics.get('current_fps')}")
    print(f"Bitrate: {metrics.get('current_bitrate')} bps")
    print(f"Frames sent: {metrics.get('frames_sent')}")
    print(f"Bytes sent: {metrics.get('bytes_sent')}")
    print(f"Viewers: {metrics.get('viewer_count')}")
    print(f"Buffer health: {metrics.get('buffer_health')}%")
else:
    print("No active stream")
```

**Safe Access Pattern** (recommended):

```python path=null start=null
from pprint import pprint

# Get status
status = await browser.get_stream_status()

# Print full structure to inspect
pprint(status)

# Safe nested access with .get() and defaults
stream_info = status.get('status', {})
metrics = stream_info.get('metrics', {})

print(f"📊 Stream Metrics:")
print(f"  Active: {status.get('active', False)}")
print(f"  State: {stream_info.get('state', 'unknown')}")
print(f"  Health: {stream_info.get('health', 'unknown')}")
print(f"  FPS: {metrics.get('current_fps', 0):.1f}")
print(f"  Bitrate: {metrics.get('current_bitrate', 0):.0f} bps")
print(f"  Viewers: {metrics.get('viewer_count', 0)}")
```

### RTMP Streaming

Stream to platforms like Twitch, YouTube, Facebook:

```python path=null start=null
# Stream to Twitch
stream = await browser.start_stream(
    protocol="rtmp",
    quality="high",
    codec="h264",
    rtmp_url="rtmp://live.twitch.tv/app",
    rtmp_key="your_stream_key_here"
)

# Stream to YouTube Live
stream = await browser.start_stream(
    protocol="rtmp",
    quality="high",
    codec="h264",
    rtmp_url="rtmp://a.rtmp.youtube.com/live2",
    rtmp_key="your_youtube_stream_key"
)
```

## Quality Profiles

### low_bandwidth
- **Target Bitrate**: 500 kbps
- **CRF**: 28
- **Use Case**: Mobile viewers, limited bandwidth
- **Latency**: ~2-3 seconds
- **File Size**: ~3.75 MB/minute

### medium (Default)
- **Target Bitrate**: 1.5 Mbps
- **CRF**: 23
- **Use Case**: Standard streaming, balanced quality/bandwidth
- **Latency**: ~2 seconds
- **File Size**: ~11.25 MB/minute

### high
- **Target Bitrate**: 3 Mbps
- **CRF**: 20
- **Use Case**: High-quality recording, fast connections
- **Latency**: ~2 seconds
- **File Size**: ~22.5 MB/minute

### lossless
- **CRF**: 0
- **Use Case**: Archival, post-processing
- **Latency**: ~3 seconds
- **File Size**: ~100+ MB/minute

## Video Codecs

### H.264 (AVC)
- **Best for**: Universal compatibility, web streaming
- **Compression**: Good (better than VP8)
- **Browser Support**: All browsers
- **Hardware Accel**: NVENC (NVIDIA), VideoToolbox (Apple), QSV (Intel)
- **Bandwidth Savings**: Baseline

### H.265 (HEVC)
- **Best for**: High-quality at lower bandwidth
- **Compression**: Excellent (30-50% better than H.264)
- **Browser Support**: Safari, Edge (limited)
- **Hardware Accel**: NVENC (NVIDIA), VideoToolbox (Apple M1+)
- **Bandwidth Savings**: 30-50% vs H.264

### VP9
- **Best for**: Open-source pipelines, YouTube-style delivery
- **Compression**: Excellent (similar to H.265)
- **Browser Support**: Chrome, Firefox, Edge
- **Hardware Accel**: Limited
- **Bandwidth Savings**: 30-40% vs H.264

### VP8
- **Best for**: Legacy systems, WebRTC
- **Compression**: Moderate (similar to H.264)
- **Browser Support**: All browsers
- **Hardware Accel**: None
- **Bandwidth Savings**: Similar to H.264

## Hardware Acceleration

FlyBrowser automatically detects and uses hardware encoders:

### NVIDIA NVENC
- Detects CUDA-capable GPUs
- Supports H.264, H.265
- Enables with `-c:v h264_nvenc` or `-c:v hevc_nvenc`
- Falls back to software if unavailable

### Apple VideoToolbox
- Detects Apple Silicon (M1/M2/M3) and Intel Macs
- Supports H.264, H.265
- Enables with `-c:v h264_videotoolbox` or `-c:v hevc_videotoolbox`
- Significantly faster than software encoding

### Intel Quick Sync (QSV)
- Detects Intel processors with integrated graphics
- Supports H.264, H.265
- Enables with `-c:v h264_qsv` or `-c:v hevc_qsv`
- Good performance for moderate quality

## CLI Commands

### Stream Management

```bash path=null start=null
# Start a stream
flybrowser stream start <session_id> \
    --protocol hls \
    --quality medium \
    --codec h264 \
    --endpoint http://localhost:8000

# Stop stream
flybrowser stream stop <session_id>

# Get stream status with metrics
flybrowser stream status <session_id>

# Get just the stream URL
flybrowser stream url <session_id>
```

### Recordings Management

```bash path=null start=null
# List all recordings
flybrowser recordings list

# List recordings for specific session
flybrowser recordings list --session-id sess_abc123

# Download a recording
flybrowser recordings download rec_xyz789 -o recording.mp4

# Delete a recording
flybrowser recordings delete rec_xyz789

# Clean old recordings (older than 7 days)
flybrowser recordings clean --older-than 7d

# Clean recordings older than 30 days
flybrowser recordings clean --older-than 30d
```

## Storage Backends

### Local Storage (Default)
```python path=null start=null
# Configured via environment
FLYBROWSER_RECORDING_STORAGE=local
FLYBROWSER_RECORDING_DIR=~/.flybrowser/recordings
FLYBROWSER_RECORDING_RETENTION_DAYS=30
```

Recordings stored at `~/.flybrowser/recordings` with metadata in `.metadata/` subdirectory.

### S3/MinIO Storage
```python path=null start=null
FLYBROWSER_RECORDING_STORAGE=s3
FLYBROWSER_S3_BUCKET=my-recordings
FLYBROWSER_S3_REGION=us-east-1
FLYBROWSER_S3_ENDPOINT_URL=https://s3.amazonaws.com  # or MinIO endpoint
FLYBROWSER_S3_ACCESS_KEY=your_access_key
FLYBROWSER_S3_SECRET_KEY=your_secret_key
FLYBROWSER_S3_PREFIX=flybrowser/recordings/
```

Features:
- Presigned URLs (24-hour expiry)
- Automatic retry on failures
- Server-side encryption support
- Lifecycle policies for automatic cleanup

### Shared/NFS Storage
```python path=null start=null
FLYBROWSER_RECORDING_STORAGE=shared
FLYBROWSER_RECORDING_DIR=/mnt/nfs/flybrowser/recordings
```

For cluster deployments with shared filesystem.

## Performance Optimization

### Bandwidth Optimization

H.265 provides the best bandwidth savings:

```python path=null start=null
# Save ~40% bandwidth vs H.264
stream = await browser.start_stream(
    codec="h265",
    quality="medium"
)
```

Expected bandwidth usage:
- H.264 medium: ~1.5 Mbps
- H.265 medium: ~900 kbps (40% savings)
- VP9 medium: ~1.0 Mbps (33% savings)

### Latency Optimization

For minimum latency:

```python path=null start=null
stream = await browser.start_stream(
    protocol="dash",      # Lower latency than HLS
    quality="low_bandwidth",  # Smaller segments
    codec="h264"          # Hardware acceleration
)
```

Expected latencies:
- DASH: <1 second
- HLS: 2-3 seconds
- RTMP: <500ms (but requires relay)

### Frame Rate Considerations

- **30 FPS**: Standard, good for most use cases
- **60 FPS**: Smooth motion, higher bandwidth (2x)
- **15 FPS**: Low bandwidth, acceptable for static content

## Cluster Deployment

In cluster mode, recordings are replicated via Raft:

```python path=null start=null
# Metadata stored in Raft for consistency
# Files stored in shared storage or S3
# Automatic failover if node goes down
# Session affinity ensures stream continuity
```

### Cluster Configuration

```bash path=null start=null
# Start cluster nodes with recording enabled
flybrowser-serve \
    --cluster \
    --node-id node1 \
    --port 8001 \
    --raft-port 4321 \
    --max-sessions 50

# Configure storage backend
export FLYBROWSER_RECORDING_STORAGE=s3
export FLYBROWSER_S3_BUCKET=cluster-recordings
```

## Deployment Modes Comparison

| Feature | Embedded | Standalone | Cluster |
|---------|----------|------------|---------|
| Recording | ✓ Local files | ✓ Configurable storage | ✓ S3/Shared storage |
| Live Streaming | ✓ Local server | ✓ Full support | ✓ Full support |
| Stream URL | localhost:random | Server URL | Load-balanced URL |
| Storage | Local only | Local/S3/NFS | S3/NFS recommended |
| Hardware Accel | ✓ | ✓ | ✓ |
| Concurrent Streams | 1 | Configurable | Auto-scaled |
| Failover | N/A | N/A | Automatic |

## Best Practices

### 1. Choose the Right Codec
- **Web streaming**: H.264 for compatibility
- **Bandwidth-limited**: H.265 or VP9
- **Open-source**: VP9
- **Archive**: H.265 lossless or high quality

### 2. Quality Profile Selection
- **Live demos**: medium quality
- **Mobile viewers**: low_bandwidth
- **Recordings for review**: high quality
- **Archival**: lossless

### 3. Storage Management
- Set retention policies (`FLYBROWSER_RECORDING_RETENTION_DAYS`)
- Use S3 lifecycle rules for automatic cleanup
- Monitor storage usage in cluster mode
- Clean old recordings regularly

### 4. Monitoring
- Check stream health regularly
- Monitor buffer health for quality issues
- Track bandwidth usage
- Alert on unhealthy streams

### 5. Hardware Acceleration
- Enable for production workloads
- Test fallback to software encoding
- Monitor GPU utilization
- Consider dedicated encoding nodes in cluster

## Troubleshooting

### FFmpeg Not Found
```bash path=null start=null
# Install FFmpeg
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Fedora/RHEL
sudo dnf install ffmpeg
```

### Stream Not Starting
Check logs for errors:
```python path=null start=null
import logging
logging.basicConfig(level=logging.DEBUG)
```

Common issues:
- FFmpeg not installed or not in PATH
- Insufficient disk space
- Port already in use (embedded mode)
- Hardware acceleration not available

### Poor Stream Quality
- Increase quality profile
- Use H.265 or VP9 codec
- Enable hardware acceleration
- Increase target bitrate
- Check network bandwidth

### High Latency
- Use DASH instead of HLS
- Reduce segment duration
- Use lower quality profile
- Enable hardware acceleration
- Check server CPU usage

## API Reference

See the [SDK Reference](../reference/sdk.md) for complete API documentation:

- `start_recording()` - Start recording session
- `stop_recording()` - Stop and save recording
- `start_stream()` - Start live stream
- `stop_stream()` - Stop stream and get statistics
- `get_stream_status()` - Get real-time stream metrics
- `list_recordings()` - List available recordings
- `download_recording()` - Download recording file
- `delete_recording()` - Delete recording

## Examples

See the [examples directory](../../examples/) for complete working examples:

- `examples/recording_basic.py` - Basic recording
- `examples/streaming_hls.py` - HLS streaming
- `examples/streaming_rtmp.py` - RTMP to Twitch/YouTube
- `examples/cluster_recording.py` - Cluster recording with S3
