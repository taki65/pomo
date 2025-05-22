import time
import win32con
import win32ui
import win32gui
from PIL import Image
import win32process
import psutil
import heapq
from collections import defaultdict
workForgData = defaultdict(lambda:{"icon":Image,"min":int,"subtitle":str})
sleepForgData = defaultdict(lambda:{"icon":Image,"min":int,"subtitle":str})
def set_foregroundstate_to_dict(state):#state ==0 :타이머정지 state ==1 작업중 state ==2 휴식중, dict에 프로그램 저장장
    if state == 0:
        return None
    info =get_active_window_info()
    if state == 1:
        if info['process_name'] in workForgData:
            workForgData[info['process_name']]["min"]+=1
        else:
            workForgData[info['process_name']] = {"icon": get_foreground_window_icon(), "min": 1, "subtitle": info['window_title']}
    elif state ==2:
        if info['process_name'] in sleepForgData:
            sleepForgData[info['process_name']]["min"]+=1
        else:
            sleepForgData[info['process_name']] = {"icon": get_foreground_window_icon(), "min": 1, "subtitle": info['window_title']}
def get_fifth_window(state): #최대 상위 5개 활성화 프로그램 정보 얻기
    if state == 1:
        top5_by_min = heapq.nlargest(5, workForgData.items(), key=lambda x: x[1]['min'])
    elif state == 2:
        top5_by_min = heapq.nlargest(5, sleepForgData.items(), key=lambda x: x[1]['min'])
    return top5_by_min
def get_foreground_window_icon(): #아이콘 이미지 얻기기
    hwnd = win32gui.GetForegroundWindow()
    if not hwnd:
        return None
    hicon = win32gui.SendMessage(hwnd, win32con.WM_GETICON, win32con.ICON_BIG, 0)
    if hicon == 0:
        hicon = win32gui.SendMessage(hwnd, win32con.WM_GETICON, win32con.ICON_SMALL, 0)
    if hicon == 0:
        hicon = win32gui.GetClassLong(hwnd, win32con.GCL_HICON)
    if hicon == 0:
        print("아이콘 핸들을 찾을 수 없습니다.")
        return None

    icon_info = win32gui.GetIconInfo(hicon)
    hbmp_color = win32ui.CreateBitmapFromHandle(icon_info[4])
    bmpinfo = hbmp_color.GetInfo()
    bmpstr_color = hbmp_color.GetBitmapBits(True)
    # Create PIL image
    img = Image.frombuffer(
        'RGBA',
        (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
        bmpstr_color,
        'raw',
        'BGRA',
        0,
        1
    )
    win32gui.DeleteObject(icon_info[4])
    return img

def get_active_window_info(): #윈도우 정보 얻기기
    hwnd = win32gui.GetForegroundWindow()  # 현재 포커스된 윈도우 핸들
    window_title = win32gui.GetWindowText(hwnd)  # 창 제목

    # 프로세스 ID 얻기
    _, pid = win32process.GetWindowThreadProcessId(hwnd)
    process = psutil.Process(pid)
    process_name = process.name()

    return {
        'window_title': window_title,
        'process_name': process_name,
    }
# while True:
    # icon_image = get_foreground_window_icon()
    # info = get_active_window_info()
    # print(f"프로세스 이름: {info['process_name']}")
    # print(f"윈도우 제목: {info['window_title']}")
    # time.sleep(1)
    # set_foregroundstate_to_dict(1)
    # print(get_fifth_window(1)[0][1]["min"])