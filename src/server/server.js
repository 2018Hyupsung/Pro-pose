const express = require('express')
const app = express()
const cors = require('cors')

app.use(cors());
app.use(express.json())
app.use(express.urlencoded({extended: true}))

const todoList = [
    {
        id: 1,
        text: '할일 1',
        done: false,
    }
];

app.get('/api/todo', function (req, res) {
  res.json(todoList);
});

app.listen(4000)
