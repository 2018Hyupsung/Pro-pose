import './App.css';
import Day from './component/Day';
import DayList from './component/DayList';
import Upload from './component/Upload';
import Main from './component/Main';
import ExercisePage from './component/exercisePage'
import Yoga from './component/yoga'
import { BrowserRouter, Routes , Route } from "react-router-dom";

function App(props) {
  

  return (
    
    <BrowserRouter>
    <Routes>
        <Route path ='/main' element={<Main />}/>
        <Route path ='/day' element={<Day />}/>
        <Route path ='/upload' element={<Upload />}/>
        <Route path ='/daylist' element={<DayList />}/>
        <Route path ='/exercisePage' element={<ExercisePage />}/>
        <Route path ='/yoga' element={<Yoga />}/>
    </Routes>
  </BrowserRouter>   
    
    
  )

}
export default App;
