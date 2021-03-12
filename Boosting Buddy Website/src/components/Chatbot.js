import React from 'react'
import axios from 'axios'
class Chatbot extends React.Component {
  state = {
    chat: [],
    msg: ''
  }
  handleChange = (e) => {
    console.log(e.target.value);
    this.setState({ msg: e.target.value });
  }
  handleSend = () => {
    if (this.state.msg !== '') {
      axios.post('https://boosterbuddy.herokuapp.com/api/talk', { 'msg': this.state.msg })
        .then(res => {
          let ch = this.state.chat;
          ch.push({ from: 'our', msag: this.state.msg })
          ch.push({ from: 'cb', msag: res.data })
          this.setState({ chat: ch, msg: '' });
          let interval = window.setInterval(function () {
            var elm = document.getElementById('chatt');
            elm.scrollTop = elm.scrollHeight;
            window.clearInterval(interval);

          }, 1000);
        })
        .catch(err => {
          console.log(err)
        });
    }
  }
  render() {
    return (
      <div>
        <div id='chatt' style={{ overflow: 'scroll', overflowX: 'hidden', height: '55vh', width: '100%' }}>
          {
            this.state.chat.map((msg) => {
              if (msg.from === 'cb') {
                return <div className="h-auto" style={{ flexWrap: 'wrap', fontSize: '20px', fontFamily: 'sans-serif', marginBottom: '10px', borderRadius: '50px', marginRight: '500px', width: '30%', backgroundColor: '#8dc498', color: 'white', float: 'left', textAlign: "center", }}>{msg.msag} </div>

              }
              else {
                return <div className="h-auto w-30 text-light float-right font-family-sans-sarif  " style={{ flexWrap: 'wrap', color: 'whitesmoke', borderRadius: '50px', fontSize: '20px', fontFamily: 'sans-serif', marginBottom: '10px', marginLeft: '500px', width: '30%', backgroundColor: '#cdd984', float: 'right', textAlign: "center" }}>{msg.msag} </div>
              }
            })
          }

        </div>
        <div style={{ height: '1vh' }}>
          <input type='text' name='msg'
            onChange={(e) => this.handleChange(e)}
            className="form-control"

            style={{ width: '85%', float: 'left' }}
            value={this.state.msg} />
          <button onClick={() => this.handleSend()} className="btn btn-secondary">Send</button>

        </div>
      </div >
    )
  }
}
export default Chatbot;