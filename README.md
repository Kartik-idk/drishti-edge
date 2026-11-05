# Drishti-Edge (दृष्टि-Edge)

> A privacy-preserving, edge-native prototype for urban safety auditing and video-scene hazard mapping.

Developed for the Seva First Innovation Challenge, Drishti-Edge explores how local computer-vision processing can support municipal safety audits without sending raw street-scene footage to the cloud.

## The idea

Conventional safety auditing is manual and infrequent; conventional surveillance can create serious privacy risks. Drishti-Edge processes video locally, keeps raw frames transient, and records structured non-identifying telemetry for an auditor dashboard.

## Core capabilities

- Local video inference using OpenCV and Ultralytics YOLO.
- Privacy-first redaction of faces and license plates before visual artifacts are produced.
- Fail-closed output behaviour when a privacy stage encounters an error.
- A local Streamlit dashboard bound to `127.0.0.1`.
- 3×3 video-scene spatial analysis for lighting, mobility exposure, and manual infrastructure inputs.
- A transparent Zone Vulnerability Index (ZVI) that reports incomplete evidence rather than silently inventing values.
- Automated tests for audit modes, privacy controls, and spatial-zone logic.

> This is a research prototype for comparative auditing—not a crime-prediction system, safety certification, or automatic road-defect detector.

## Stack

Python 3.10+, OpenCV, Ultralytics YOLO, Streamlit, SQLite, and python-dotenv.

## Quick start

```bash
git clone https://github.com/Kartik-idk/drishti-edge.git
cd drishti-edge
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Place permitted local test clips in `data/videos/`. Raw source videos are ignored by Git and should never be committed.

## Test

```bash
./.venv/bin/python -m unittest discover tests
```

## Privacy commitments

- No cloud video pipeline or external transmission.
- Raw frames are processed in memory and discarded.
- Identifiable visual artifacts are redacted before output.
- Input source videos are not overwritten.
- Fail-closed behaviour protects against outputting unredacted content after an error.

## Further reading

- [Benchmark reports](docs/benchmarks/)
- [Third-party notices](THIRD_PARTY_LICENSES.md)
- [Environment template](.env.example)
