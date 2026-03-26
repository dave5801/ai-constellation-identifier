# AI Constellation & Object Identifier

Production-quality monorepo for uploading a night-sky photo, detecting stars, identifying supported constellations, scoring confidence, and returning an annotated image.

## Project Structure

```text
ai-constellation-identifier/
  backend/
    data/
    vision/
    main.py
    requirements.txt
    Dockerfile
  frontend/
    src/
    package.json
  README.md
```

## Features

- FastAPI backend with image upload, preprocessing, blob-based star detection, catalog-based star-field matching, possible bright-object detection, and base64 annotated image output
- React + TypeScript + Tailwind frontend with drag-and-drop upload, image preview, responsive layout, loading state, confidence bars, and annotated result display
- Local-development CORS enabled for Vite and common localhost ports
- Optional backend Dockerfile and `.env.example` files for local configuration

## Backend Setup

1. Create a Python 3.10+ virtual environment.

```bash
cd backend
python -m venv .venv
```

2. Activate the virtual environment.

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

4. Run the FastAPI server.

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

5. Verify health.

Open [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) and confirm:

```json
{"status":"ok"}
```

## Frontend Setup

1. Open a new terminal and move into the frontend.

```bash
cd frontend
```

2. Install dependencies.

```bash
npm install
```

3. Optionally copy `.env.example` to `.env` and adjust the API URL if your backend is not running on `127.0.0.1:8000`.

4. Start the Vite development server.

```bash
npm run dev
```

5. Open the app in your browser.

Use the local URL printed by Vite, typically [http://127.0.0.1:5173](http://127.0.0.1:5173).

## API

### `GET /health`

Returns:

```json
{
  "status": "ok"
}
```

### `POST /identify`

Accepts multipart form data with an image file under the `file` field.

Example response:

```json
{
  "constellations": [
    {
      "name": "Orion",
      "confidence": 0.91
    }
  ],
  "stars_detected": 120,
  "possible_planets": [
    {
      "x": 442.1,
      "y": 218.5,
      "brightness": 248.4
    }
  ],
  "annotated_image": "<base64-png>"
}
```

## Detection Pipeline

1. Convert uploaded images to grayscale
2. Apply Gaussian blur and CLAHE contrast enhancement
3. Isolate bright points with thresholding and morphological cleanup
4. Detect stars with OpenCV `SimpleBlobDetector`
5. Flag unusually bright large blobs as possible planets
6. Project a local bright-star catalog for Orion, Ursa Major, Cassiopeia, and Scorpius into a normalized sky field using `astropy`
7. Seed candidate alignments with triangle-signature matching and affine fitting
8. Score alignments using geometric residuals, catalog coverage, and brightness consistency with `scikit-learn` nearest-neighbor matching
9. Return structured JSON and an annotated PNG overlay

## Production Notes

- The backend uses in-memory processing only and does not persist uploaded images.
- Matching is catalog-based rather than full plate-solving; best results come from clear images focused on one supported constellation region.
- Backend scientific dependencies now include `astropy` for sky-coordinate projection and `scikit-learn` for nearest-neighbor match scoring.
- For deployment, pin dependency versions and serve the built frontend from a static host or reverse proxy.

## Optional Docker Run

```bash
cd backend
docker build -t ai-constellation-backend .
docker run --rm -p 8000:8000 ai-constellation-backend
```
