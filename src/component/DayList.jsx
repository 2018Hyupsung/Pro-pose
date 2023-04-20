import React, {useState} from "react";
import Day from './Day';

const DayList = (props) => {
    const [email, setEmail] = useState('');
    const [pass, setPass] = useState('');
    const [id, setId] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        fetch('http://localhost:8080/members/new',{
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                id,
                pass,
                email
            })
        }).then((res)=> {if(res.ok){
            alert("회원가입 완료.")
            document.location.href = "./day";    
        }else if(!res.ok){
            alert("이미 존재하는 아이디입니다.")
        }});
    };

    return (
        <div className="App">
        <div className="WOW">
            <h2>Register</h2>
        <form className="register-form" onSubmit={handleSubmit}>
            <label htmlFor="name">Full name</label>
            <input value={id} name="id" onChange={(e) => setId(e.target.value)} id="id" placeholder="full id" />
            <label htmlFor="email">email</label>
            <input value={email} onChange={(e) => setEmail(e.target.value)}type="email" placeholder="youremail@gmail.com" id="email" name="email" />
            <label htmlFor="password">password</label>
            <input value={pass} onChange={(e) => setPass(e.target.value)} type="password" placeholder="********" id="password" name="password" />
            <button type="submit">submit</button>
            <mybutton className="link-btn"><a href="./day">Already have account?</a></mybutton>
        </form>
    </div>
    </div>
    )
}
export default DayList;