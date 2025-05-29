from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
import psutil
import asyncio
import sys
import os
workTime, breakTime,recWorkTime=0,0,0
app =FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static",StaticFiles(directory="build/static"),name="static")
@app.get("/")
async def start_path():
    return FileResponse("build/index.html")
@app.get("/{full_path:path}")
async def other_path(full_path:str):
    return FileResponse(os.path.join("build","index.html"))
# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     client = websocket
#     try:
#         while True:
#             data = await websocket.receive_json()
#             await get_time(data["worktime"],data["breaktime"])
            
#     except WebsocketDisconnect:
#         pass
class Data(BaseModel):
    workTime: int
    breakTime: int
@app.post("/get-time")
async def start_task(data: Data):
    global workTime,breakTime
    workTime = data.workTime
    breakTime = data.breakTime
    try:
        if psutil.pid_exists(pid):
            print("ISExisit!!!!!!!!!!")
            process.terminate()
            process.kill()
        else:
            print("Not Exisits")
    except:
        print("ERROR!!!!!!!!!!!!!")
        pass
    result = await asyncio.to_thread(get_time)
    #result = await get_time(workTime,breakTime)
    print("get",result)
    if result =="null":
         return {"workTime": "None"}
    return {"workTime": result}
def get_time():
    try:
        command = [sys.executable,"session_core.py",str(workTime/60),str(breakTime/60)]
        global pid,process
        process = subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
        pid = process.pid
        # for line in iter(process.stdout.readline,''):
        #     lastOutput = line.strip()
        # await process.wait()
        stdout, stderr = process.communicate()
        print(stdout)
        target_lines = [
            line for line in stdout.splitlines()
            if "추천" in line]
        for line in target_lines:
            staridx = line.find(':')
            if (staridx != -1):
                endidx = line.find('분')
                if (endidx != -1):
                    result = line[staridx+1:endidx].strip()
                    print("출력됨",result)
                    return (round(float(result),3))
        return None
    except:
        process.kill()
        return None

