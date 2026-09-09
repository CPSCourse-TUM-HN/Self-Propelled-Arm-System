import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

def download_dataset():
    rf = Roboflow(api_key=os.getenv("PRIVATE_ROBOFLOW_KEY"))
    workspace = rf.workspace(os.getenv("WORKSPACE_ID"))
    project = workspace.project(os.getenv("PROJECT_ID"))
    
    dataset = project.version(1).download(model_format="yolov11", location="./paper_detect")
    
    print("Dataset downloaded successfully to:", dataset.location)

if __name__ == "__main__":
    download_dataset()