
import sys
import shutil
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from pyspark.sql import SparkSession
from backend.inference import predict_execution



PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
INFERENCE_OUTPUT_DIR = PROJECT_ROOT / "inference_output"

EXPECTED_FILES = {
    "ata_read.csv",
    "ata_write.csv",
    "mem_exec.csv",
    "mem_read.csv",
    "mem_readwrite.csv",
    "mem_write.csv"
}



app = FastAPI(
    title="RanSMAP API",
    description="Ransomware detection using LSTM models",
    version="1.0.0"
)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



spark = (
    SparkSession.builder
    .appName("RanSMAP-Inference")
    .master("local[*]")
    .getOrCreate()
)




@app.get("/")
def home():
    return {
        "message": "RanSMAP API is running",
        "status": "success"
    }




@app.get("/health")
def health():
    return {
        "status": "healthy"
    }




@app.post("/predict")
async def predict(

    ata_read: UploadFile = File(
        ...,
        description="Upload ata_read.csv"
    ),

    ata_write: UploadFile = File(
        ...,
        description="Upload ata_write.csv"
    ),

    mem_exec: UploadFile = File(
        ...,
        description="Upload mem_exec.csv"
    ),

    mem_read: UploadFile = File(
        ...,
        description="Upload mem_read.csv"
    ),

    mem_readwrite: UploadFile = File(
        ...,
        description="Upload mem_readwrite.csv"
    ),

    mem_write: UploadFile = File(
        ...,
        description="Upload mem_write.csv"
    )

):

    files = [
        ata_read,
        ata_write,
        mem_exec,
        mem_read,
        mem_readwrite,
        mem_write
    ]


    received_files = {
        Path(file.filename).name
        for file in files
        if file.filename
    }

    if received_files != EXPECTED_FILES:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Invalid CSV files.",
                "expected_files": sorted(EXPECTED_FILES),
                "received_files": sorted(received_files)
            }
        )

  

    execution_path = Path(
        tempfile.mkdtemp(prefix="ransmap_")
    )

    try:


        for file in files:

            filename = Path(file.filename).name
            destination = execution_path / filename

            with open(destination, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)



        INFERENCE_OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )


        results = predict_execution(
            spark=spark,
            execution_path=str(execution_path),
            out_root=str(INFERENCE_OUTPUT_DIR),
            artifacts_dir=str(ARTIFACTS_DIR)
        )


        predictions = {}

        for fraction, result in results.items():

            percentage = f"{int(fraction * 100)}%"

            predictions[percentage] = {
                "probability": result["probability"],
                "prediction": result["prediction"],
                "label": (
                    "Ransomware"
                    if result["prediction"] == 1
                    else "Benign"
                )
            }

        return {
            "status": "success",
            "predictions": predictions
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:


        for file in files:
            await file.close()

        shutil.rmtree(
            execution_path,
            ignore_errors=True
        )



if __name__ == "__main__":

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
