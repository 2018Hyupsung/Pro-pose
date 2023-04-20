import React from 'react';
import { Navbar, Nav, Form, FormControl, Button } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';

const AFMenuBar = () => {
    return (
        <div className="Navigation">
    <Navbar bg="dark" variant="dark">
    <div className="move-right"><Navbar.Brand href="#home">Navbar</Navbar.Brand></div>S
    <Nav className="mr-auto">
      <Nav.Link href="./Main">Home</Nav.Link>
      <Nav.Link href="#features">Guide</Nav.Link>
      <Nav.Link href="./day">Logout</Nav.Link>
    </Nav>
    <Form inline>
      <FormControl type="text" placeholder="Search" className="mr-sm-2" />
      <Button variant="outline-info">Search</Button>
    </Form>
 </Navbar>
 </div>
        
    );
};

export default AFMenuBar;