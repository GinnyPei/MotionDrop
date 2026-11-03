# MotionDrop interactive streaming demo

## Direct access from another machine

On a trusted network, start the public entry point:

```bash
./scripts/run_streaming_demo_public.sh
```

Enter the server's `sudo` password once so the lightweight port proxy can listen
on port 80. The model process continues to run as the current user. The
script prints a URL such as `http://SERVER_IP`; open it from another machine
on the same routable network. Press `Ctrl+C` to stop both processes.

To use a non-privileged public port without `sudo`:

```bash
MOTIONDROP_DEMO_PUBLIC_PORT=8080 ./scripts/run_streaming_demo_public.sh
```

Start the headless web server:

```bash
./scripts/run_streaming_demo.sh
```

The service binds to `127.0.0.1:7860` by default. From your laptop, forward
that port over SSH:

```bash
ssh -N -L 7860:127.0.0.1:7860 USER@SERVER_IP
```

Then open <http://127.0.0.1:7860> in the laptop browser. The browser plays the
video and reports its current timestamp; the server decodes the video once,
front to back, and incrementally compresses and prefills windows into one KV
history. Messages are routed automatically (or pick a type in the selector):

| Type | Example | Behaviour |
|---|---|---|
| Question | 现在比分多少？ | Answered right away; preempts running commentary |
| Standing task | 解说这场乒乓球比赛 | A new line every 4 s as a subtitle and on the commentary track |
| Watch task | 有人得分时提醒我 | Checked every second; raises an alert when it holds |
| Update | 解说的时候多关注发球 | Adds a requirement to the task it refers to |
| Stop | 停止解说 | Cancels the task it refers to (or use the task's Stop button) |

Updates stream to the page over server-sent events (`/api/events`). Seeking
backwards starts a new session.

To switch videos without restarting, paste an absolute path into the box above
the player and press 加载 (a StreamingBench sample directory containing
`video.mp4` also works). This closes the current session. Pass
`--video-root DIR` (repeatable) to restrict which directories can be loaded,
which is recommended whenever the demo is exposed beyond localhost.

Common overrides:

```bash
MOTIONDROP_STREAM_VIDEO=/path/to/video.mp4 \
MOTIONDROP_DEMO_PORT=8080 \
MOTIONDROP_FIXED_COMPRESSION=0.9 \
MOTIONDROP_MAX_CONTEXT_TOKENS=16000 \
MOTIONDROP_STANDING_INTERVAL=4 \
  ./scripts/run_streaming_demo.sh
```

Set `MOTIONDROP_LATENCY_MODEL=streaming_runtime/profiles/llava-ov-7b.json` to
size each window with the SLO controller instead of a fixed compression ratio
(it keeps far more tokens per window, so the context fills sooner).
`MOTIONDROP_MEMORY_INTERVAL=5` adds background note-taking so questions about
evicted video can still be answered from text notes.

The demo has no authentication. Use the public entry point only on a trusted
network. For untrusted networks, keep the loopback binding and use SSH
forwarding.
