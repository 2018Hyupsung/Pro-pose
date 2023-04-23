import React, { useRef, useEffect } from 'react';
import { BsGithub, BsYoutube } from 'react-icons/bs';
import 'bootstrap/dist/css/bootstrap.min.css';
import styles from './Css/Footer.module.scss';


const Footer = () => {
    return(
        <footer className={styles.footer_set}>
            <div className={styles.number1}>
                <div className={styles.number1_1}>
                    ABOUT
                </div>
                <div className={styles.number1_2} style={{textAlign: 'center'}}>
                    서비스명 : 프로,포즈(Pro,pose)<br/>
                    지도교수 : 홍석주 | 팀원:김봉준,서재원,장준영<br/>
                    이메일 : 2018hyupsung@gmail.com<br/>
                    주소 : 화성시 봉담읍 최루백로 72
                </div>
            </div>

            <div className={styles.number2}>
                <div className={styles.number1_1}>
                    LICENSE
                </div>
                <div className={styles.number1_2} style={{textAlign: 'center'}}>
                    This service is licensed under the Apache License, Version 2.0.
                </div>
            </div>

            <div className={styles.number3}>
                <div className={styles.number1_1}>
                    CONTACT
                </div>
                <div className={styles.number1_2} style={{textAlign: 'center'}}>
                    <br/>
                    <a href="https://www.github.com/2018hyupsung/pro-pose">
                        <BsGithub size={50} style={{color: 'white'}}/>
                    </a>
                    <a style={{marginLeft:'20px'}}></a>
                    <a href="https://youtube.com">
                        <BsYoutube size={50} style={{color: 'white'}}/>
                    </a>
                </div>
            </div>

            <div className={styles.number5}>
                Copyright ⓒ 2023. Pro,pose All Right Reserved.
            </div>
        </footer>
    );
};


export default Footer;