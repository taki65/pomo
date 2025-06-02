import react, { useEffect, useRef, useState } from "react";
import axios from 'axios'
import { atom, useAtom, useAtomValue } from "jotai";
import timercss from './Timer.module.scss';
import './static.scss';
export const timerAtom = atom(0);
export const timerOnAtom = atom(false, (get, set) => set(timerOnAtom, !get(timerOnAtom)));
export const timerStateAtom = atom(false, (get, set) => set(timerStateAtom, !get(timerStateAtom))); // false이면 work, true이면 break

export function Timer() {
  const [isGetTime,setIsGetTime] = useState(true);
  //const [times, setTimes] = useState({workTime:5,breakTime:5});
  const [times, setTimes] = useState(() =>{
    const times = localStorage.getItem("times");
    return times? JSON.parse(times) : {workTime:60,breakTime:60}
  });
  const [timerState, setTimerState] = useAtom(timerStateAtom);
  const [timerOn, setTimerBool] = useAtom(timerOnAtom);
  const [time, setTime] = useAtom(timerAtom);
  const frameRef = useRef();
  const sec = useRef(0);
  const standSec = useRef(0);
  const GetTimeRef = useRef({isGetTime:true,workTime:times.workTime,breakTime:times.breakTime,isRunning:false});
  const resetFunc = () => {
    let initTime; if (timerState) {
      initTime = times.breakTime * 1000;
    } else {  
      initTime = times.workTime * 1000;
    } if (timerOn) {

      setTimerBool();
      setTime(initTime);
      sec.current = initTime;
      setTimeout(setTimerBool, 0);
    }
    else {
      setTime(initTime);
      sec.current = initTime;
    }
  }
  useEffect(() =>{ // 처음 초기화
    const interval = setInterval(() =>{
      console.log("Interval",GetTimeRef.current.isGetTime);
          if (GetTimeRef.current.isGetTime && !GetTimeRef.current.isRunning){
            GetTimeRef.current.isRunning=true;
        (async () =>{
          try{
            const responce = await axios.post("http://localhost:8000/get-time",{workTime:GetTimeRef.current.workTime,breakTime:GetTimeRef.current.breakTime})
            if(GetTimeRef.current.isGetTime){
                 console.log("getvalue:",responce.data.workTime);
                 if (responce.data.workTime != "None"){
                  const get_time=parseFloat(responce.data.workTime);
                  console.log("getFin",get_time);
                 setTimes((prev) =>({...prev,workTime:Math.floor(get_time*60)}));
                 }
            }
          }
          catch(error){
            console.error(error);
          }
          GetTimeRef.current.isRunning=false;
        })();
        
      }
    },5000);

    return () =>clearInterval(interval);
  },[])
  useEffect(() => { // 타이머가 시작되거나 중지될 때마다 실행
    if (!timerOn) {
      return;
    }
    standSec.current = performance.now();
    const update = () => {
      const newtime = sec.current - (performance.now() - standSec.current);
      setTime(newtime);
      if (newtime <= 0) { // 타이머가 0이 되면 상태 변경
        if (timerState){
          document.documentElement.style.setProperty("--main-color", "tomato");
        }
        else{
document.documentElement.style.setProperty("--main-color", "#333333");
        }
        setTimerState();
        return;
      }
      frameRef.current = requestAnimationFrame(update);
    }
    frameRef.current = requestAnimationFrame(update);
    return () => {
      cancelAnimationFrame(frameRef.current);
    }
  }, [timerOn])
  useEffect(() => {  // 타이머 상태가 변경될 때마다 실행
    resetFunc();
  }, [timerState])
  useEffect(() => {
    GetTimeRef.current.workTime = times.workTime;
    GetTimeRef.current.breakTime = times.breakTime;
    localStorage.setItem("times",JSON.stringify(times));
    if (time > (timerState ? times.breakTime * 1000 : times.workTime * 1000)) {
      resetFunc();
    }
  }, [times.workTime, times.breakTime]); // 작업 시간과 휴식 시간이 변경될 때마다 실행
  return (
    <div className={timercss.parent1}>
      < CircleProgressBar className={`${timercss.circleProgressBar} ${(time<30000 && timerOn)? timercss.shake:""}`} time={Math.floor(time)} maxtime={timerState ? times.breakTime * 1000 : times.workTime * 1000} timerState={timerState}></CircleProgressBar>
      <h1>남은시간 : {Math.floor(time / 60000)}분 {Math.floor((time % 60000) / 1000)}초</h1>
      <div className={timercss.parent2}>
          
      <button className={timercss.popBtn} onClick={() => {
        if (!timerOn) {
          sec.current = time;
        } setTimerBool()
      }}>{timerOn ? "Stop" : "Start"}</button>
      <button className={timercss.popBtn} onClick={resetFunc}>Reset</button>
      </div>
            <input className={timercss.input} value={times.workTime} onChange={(e) => setTimes((prev)=>({...prev,workTime:e.target.value}))} placeholder="작업 시간(초)"></input><>work : {Math.floor(times.workTime / 60)}분 {times.workTime % 60}초</>
      <input className={timercss.input} value={times.breakTime} onChange={(e) => setTimes((prev)=>({...prev,breakTime:e.target.value}))} placeholder="휴식 시간(초)"></input><>break : {Math.floor(times.breakTime / 60)}분 {times.breakTime % 60}초</>
      <div>
        <>추천시간</>
<input type="checkbox" checked={isGetTime} onChange={(e) =>{setIsGetTime(prev =>{
GetTimeRef.current.isGetTime=!prev;
  return e.target.checked});}}></input>
      </div>
      
    </div>
  );
}
export function CircleProgressBar({ time, maxtime, timerState,className }) {
  const radius = 80;
  const progress = (time / maxtime) * 360 - 0.0001;
  const radian = (progress - 90) * (Math.PI / 180);
  const x = radius + radius * Math.cos(radian);
  const y = radius + radius * Math.sin(radian);
  const largeArcFlag = progress > 180 ? 1 : 0;
  const pathData = `M ${radius} ${radius} L ${radius} ${0} A ${radius} ${radius} 0 ${largeArcFlag} 1 ${x} ${y} Z`;
  return (
    <>
      <svg className={className} width="160" height="160">

        <circle cx={80} cy={80} r={80} fill={!timerState ? "#333333" : "tomato"} />
        <path d={pathData} fill={timerState ? "#333333" : "tomato"} />
      </svg>

    </>

  )
}