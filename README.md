# Smart Search – Simplified Deploy

Video-based object/person detection with natural language queries.  
Camera input removed. Only video file upload is supported.

## What's included
| Module | Purpose |
|---|---|
| `app.py` | Flask API + auth + video processing |
| `detector.py` | YOLOv8 detection & tracking |
| `query_parser.py` | Color & clothing attribute parsing |
| `attribute_analyser.py` | Color region analysis on crops |
| `matcher.py` | Attribute + keyword matching |
| `nlp_engine.py` | Natural language query interpretation |
| `alerter.py` | Draw bounding boxes on frames |

## Removed (not needed)
- OpenCV camera (source=0)
- deepface / face-recognition
- torchreid
- CLIP (openai)
- MongoDB
- Google Auth

## Local run

```bash
pip install -r requirements.txt
python app.py
# visit http://localhost:5000
```

Default login: `admin` / `admin`

## Deploy to Render.com (recommended)

1. Push this folder to a new GitHub repo
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your repo
4. Build command: `pip install -r requirements.txt`
5. Start command: `python app.py`
6. Done ✓

> **Note:** The `yolov8s.pt` model (~22 MB) is included in the repo.  
> On first deploy it downloads automatically if missing via ultralytics.

## Deploy to Railway

1. Push to GitHub
2. Create new project on [railway.app](https://railway.app)
3. Connect repo → it auto-detects Python
4. Set `PORT=5000` environment variable if needed

## Usage

1. Log in at `/`
2. Upload a video file
3. Enter a query like:
   - `person with red shirt`
   - `blue car`
   - `person running`
   - `count people`
4. Click **Start** — matched frames stream live with bounding boxes
5. Detections are saved to `/reports/` and viewable in the Reports page
