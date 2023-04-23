import React, { useState, useEffect } from 'react';
import { Navbar, Nav, Container} from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import styles from './Css/Menubar.module.scss';


const MenuBar = () => {
  const [navbarStyle, setNavbarStyle] = useState({
    backgroundColor: 'transparent',
    boxShadow: 'none',
    transition: 'all 0.3s ease-in-out'
  }); // 초기값은 투명색과 그림자 없음
  
  const [Image, setImage] = useState('Logo/Logo_black_bar.svg');

  
  useEffect(() => {
    const handleScroll = () => {
      const currentScrollY = window.scrollY;
      const navbarHeight = document.querySelector('nav').offsetHeight;
      if (currentScrollY > navbarHeight) {
        setNavbarStyle({
          backgroundColor: 'rgba(255,255,255,1)',
          boxShadow: '0px 3px 22px 1px #555555', // 그림자 효과 추가
          transition: 'all 0.3s ease-in-out'
        }); // 스크롤 위치에 따라 Navbar 스타일 변경
        
        setImage('Logo/Logo_color_bar.svg');

      } else {
        setNavbarStyle({
          backgroundColor: 'transparent',
          boxShadow: 'none', // 그림자 없애기
          transition: 'all 0.3s ease-in-out'
        });
        
        setImage('Logo/Logo_black_bar.svg');
      }
    };
    window.addEventListener('scroll', handleScroll);
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  return (
    <Navbar style={navbarStyle} fixed='top'>
      <Container>
        <div>
          <Navbar.Brand href='./Main'>
            <img src={Image} className={styles.logo}/>
          </Navbar.Brand>
        </div>
        <Nav>
          <Nav.Link href="./Main" className={styles.bar_line}>메인</Nav.Link>
          <Nav.Link href="./day" className={styles.bar_line}>로그인</Nav.Link>
          <Nav.Link href="./daylist" className={styles.bar_line}>회원가입</Nav.Link>
        </Nav>
      </Container>
    </Navbar>
  );
};

export default MenuBar;