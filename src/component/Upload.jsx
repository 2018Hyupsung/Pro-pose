import React from 'react';
import {useState,useRef} from 'react';
import MenuBar from './MenuBar';
import { Navbar, Nav, Form,Button,Input,Container } from 'react-bootstrap';
import ReactPlayer from 'react-player';

const Upload = () => {  
    const fileInput = React.useRef(null);

    const handleButtonClick = e =>{
        fileInput.current.click();
    };

    const handleChange = e =>{
        console.log(e.target.files[0]);
    };

    return (
  <div>
    <MenuBar />
    <React.Fragment>
    <Button onClick={handleButtonClick}>파일 업로드</Button>
    <input type="file" accpet="video/mp4" ref={fileInput} onChange={handleChange} style={{display:"none"}}/>
    </React.Fragment>
    
  </div>
  
    
    );
  }
  export default Upload;