import React from 'react';
import MenuBar from './MenuBar';
import { useEffect, useState } from 'react';

const Yoga = () => {  
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
      <h1>요가 페이지 입니다.</h1>
        <button>운동 시작</button>
    </div>  
  );
}

export default Yoga;
  