import React from 'react';
import MenuBar from './MenuBar';
import MenuBarBeforeLogin from './MenuBarBeforeLogin'
import { useEffect, useState } from 'react';
import axios from 'axios';


const Main = () => {  
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

  if(id == null){
    return (
      <div>
        <MenuBarBeforeLogin />
        <h1>로그인 전 메인 페이지 입니다.</h1>
    
      </div>
    )
  }
  else{
    return (
      <div>
        <MenuBar />
        <h1>로그인 후 메인 페이지 입니다.</h1>
          <div>{id}</div>
          <div>{pass}</div>
          <div>{email}</div>
      </div>  
    );
  }
  
  }
  export default Main;
  