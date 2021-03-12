import React, { Component } from 'react';


class About extends Component {
  constructor(props) {
    super(props);
    this.state = {
      aboutItems: [
        { id: 1, delay: "100", iconClass: "bi bi-chat-square-text", linkURL: "/#", title: "AI Chatbot", description: "Boosting Buddy use Deep Learning models for training. Natural Language Processing is also used. " },
        { id: 2, delay: "200", iconClass: "bi bi-tablet", linkURL: "/#", title: "Web/Mobile Application", description: "Bossting Buddy mobile application as well as website application is available 24/7 for users to talk anytime. " },
        { id: 3, delay: "300", iconClass: "bi bi-people", linkURL: "/#", title: "Roman Urdu", description: "Boosting Buddy is a Roman Urdu chatbot which is very helpful for the local comunity of Pakistan, who are suffering from mental diseases." }
      ]
    }
  }


  AboutItem(item) {
    return (
      <div className="icon-box" data-aos="fade-up" data-aos-delay={item.delay} key={item.id}>
        <div className="icon"><i className={item.iconClass}></i></div>
        <h4 className="title"><a href={item.linkURL}>{item.title}</a></h4>
        <p className="description">{item.description}</p>
      </div>
    );
  }
  render() {
    return (
      <>
        {this.state.aboutItems.map(item => (
          this.AboutItem(item)
        ))}
      </>
    );
  }
}

export default About;