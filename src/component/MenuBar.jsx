import React from 'react';
import { Navbar, Nav} from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';

const MenuBar = () => {
    return (
      <div>
        <div className="Navigation">
          <Navbar bg="dark" variant="dark">
          <div className="move-right" style={{margin:'auto'}}><Navbar.Brand href="./Main">Pro-Pose</Navbar.Brand></div>
          <Nav className="mr-auto" style={{margin:'auto'}}>
            <Nav.Link href="./Main" style={{width:'100px'}}>Home</Nav.Link>
            <Nav.Link href="./Upload" style={{width:'100px'}}>Upload</Nav.Link>
            <Nav.Link href="./Main" style={{width:'100px'}} onClick={()=>{localStorage.clear()}}>LogOut</Nav.Link>
          </Nav>
          </Navbar>
        </div>

        <div className="Navigation">
        <Navbar bg="dark" variant="dark">
        <Nav className="mr-auto" style={{margin:'auto'}}>
          <Nav.Link href="./exercisePage" >재활 운동 하기</Nav.Link>
          <Nav.Link href="./Upload" >운동 일지</Nav.Link>
          <Nav.Link href="./Main" >커뮤니티</Nav.Link>
          <Nav.Link href="./Main" >마이페이지</Nav.Link>
        </Nav>
        </Navbar>
      </div>
    </div>
        
    );
};

export default MenuBar;