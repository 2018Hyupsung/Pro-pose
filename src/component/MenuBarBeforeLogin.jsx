import React from 'react';
import { Navbar, Nav} from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';

const MenuBar = () => {
    return (
      <div className="Navigation">
        <Navbar bg="dark" variant="dark">
          <div className="move-right" style={{margin:'auto'}}><Navbar.Brand href="./Main">Pro-Pose</Navbar.Brand></div>
          <Nav className="mr-auto" style={{margin:'auto'}}>
            <Nav.Link href="./Main" style={{width:'100px'}}>Home</Nav.Link>
            <Nav.Link href="./day" style={{width:'100px'}}>Login</Nav.Link>
            <Nav.Link href="./daylist" style={{width:'100px'}}>Register</Nav.Link>
          </Nav>
        </Navbar>
      </div>
        
    );
};

export default MenuBar;