import React from 'react';
import MenuBar from './MenuBar';
import MenuBarBeforeLogin from './MenuBarBeforeLogin'
import { useEffect, useState } from 'react';
import axios from 'axios';


const ExercisePage = () => {  
  // const [todoList, setTodoList] = useState(null);
  const [id, setId] = useState(null);
  const [pass, setpass] = useState(null);
  const [email, setemail] = useState(null);

  // const fetchData = async () => {
  //   const response = await axios.get('http://localhost:8080');
  //   setTodoList(response.data);
  // };
  useEffect(() => {
    // fetchData();
    setId(localStorage.getItem("id"));
    setpass(localStorage.getItem("pass"));
    setemail(localStorage.getItem("email"));
  }, []);
  
  // if(!todoList){
  //   return <div>loading...</div>
  // }

  return (
    <div>
      <MenuBar />
      <h1>재활 운동 페이지 입니다.</h1>
        <button onClick={()=>{document.location.href = './yoga'}} style={{width: '100px'}}>요가</button>
        <button style={{width: '100px'}}>태권도</button>
        <button style={{width: '100px'}}>???</button>
    </div>  
  );
}

export default ExercisePage;
  