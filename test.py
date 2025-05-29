import subprocess
import sys
def get_time():
    # try:
    #     command = [sys.executable,"session_core.py","1","1"]
    #     process = subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    #     for line in iter(process.stdout.readline,''):
    #         lastOutput = line.strip()
    #     process.wait()
    #     print(lastOutput)
    #     staridx = lastOutput.find(':')
    #     if (staridx != -1):
    #         endidx = lastOutput.find('분')
    #         result = lastOutput[staridx:endidx]
    #         print("결과",result)
    #         return result
    # except subprocess.TimeoutExpired:
    #     process.kill()
    #     return None
    try:
        command = [sys.executable,"session_core.py",str(1),str(1)]
        process = subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
        # for line in iter(process.stdout.readline,''):
        #     lastOutput = line.strip()
        # await process.wait()
        stdout, stderr = process.communicate()
        target_lines = [
            line for line in stdout.splitlines()
            if line.find('결과 :')
        ]
        for line in target_lines:
            staridx = line.find(':')
            if (staridx != -1):
                endidx = line.find('분')
                result = line[staridx:endidx]
                print("출력 분",result)
                return result
    except:
        process.kill()
        return None
get_time()