import React, { useRef, useEffect } from 'react';
import { Button } from 'react-bootstrap';
import { BsBoxArrowUpRight } from 'react-icons/bs';
import { gsap } from "gsap";
import 'bootstrap/dist/css/bootstrap.min.css';
import styles from './Css/Main_front.module.scss';



const Main_front = () => {
    const updown1 = useRef();
    const updown2 = useRef();
    const updown3 = useRef();
    const updown4 = useRef();

    useEffect(() => {
        const tl = gsap.timeline({ repeat: -1, yoyo: true });
        tl.to(updown1.current, {
          y: "+=23",
          duration: 2,
          ease: "power2.inOut",
        });
    }, []);

    useEffect(() => {
        const tl = gsap.timeline({ repeat: -1, yoyo: true });
        tl.to(updown2.current, {
          y: "-=40",
          duration: 2.7,
          ease: "power2.inOut",
        });
    }, []);

    useEffect(() => {
        const tl = gsap.timeline({ repeat: -1, yoyo: true });
        tl.to(updown3.current, {
          y: "+=17",
          duration: 3.0,
          ease: "power2.inOut",
        });
    }, []);
    
    useEffect(() => {
        const tl = gsap.timeline({ repeat: -1, yoyo: true });
        tl.to(updown4.current, {
          y: "+=40",
          duration: 4.0,
          ease: "power2.inOut",
        });
    }, []);

    return (
        <div className={styles.total_box}>
            <div className={styles.left_box}>
                <div className={styles.left_box_first}>
                    <span className={styles.span_button}>인공지능 기반 재활도움 솔루션</span>
                    <p className={styles.copywriting}><font style={{color:'rgb(202,56,100)'}}>프로</font>가 보여주는 포즈,<br/>프로가 잡아주는 <font style={{color:'rgb(51,110,217)'}}>포즈</font></p>
                    <br/><br/>
                    <Button variant="secondary" className={styles.button}>강의실 바로가기 <BsBoxArrowUpRight></BsBoxArrowUpRight></Button>{' '}
                </div>
            </div>

            <div className={styles.right_box}>
                <div className={styles.right_box_first}>
                    <img className={styles.character} src = 'images/Character.png' ref={updown1}/>
                    <img className={styles.landmark} src='images/Landmark.png' ref={updown2}/>
                    <img className={styles.bubble} src='images/Bubble.png' ref={updown3}/>
                    <img className={styles.python} src='images/Python.png' ref={updown4}/>
                </div>
            </div>
        </div>



    );
};

export default Main_front;