import React, { Component } from 'react';
class Logo extends Component {
  constructor(props) {
    super(props);
    this.state = {
      imgURL: "assets/img/bblogo2.png"
    }
  }
  render() {
    return (
      <div id="logo">
        <a href="index.html"><img src={this.state.imgURL} alt="" /></a>
      </div>
    );
  }
}

export default Logo;