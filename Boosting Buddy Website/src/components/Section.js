import React, { Component } from 'react';
import About from './About'
import Team from './Team'
import Contact from './Contact'
import Chatbot from './Chatbot.js'

class Section extends Component {
  constructor(props) {
    super(props);
    this.state = {
      sections: [
        { id: 0, sectionID: "hero", dataAOS: "zoom-in", dataAosDelay: "100", sectionTitle: "Boosting Buddy", sectionDescription: "A Conversational Agent for Improvement of Mental Health", btnText: "Chat Now" },
        { id: 1, comp: <Chatbot />, sectionID: "chatbot", dataAOS: "fade-up", dataAosDelay: "100", sectionCLass: "", sectionTitle: "Chatbot", sectionDescription: "" },
        { id: 2, comp: <About />, sectionID: "about", dataAOS: "fade-up", dataAosDelay: "100", sectionCLass: "", sectionTitle: "About", sectionDescription: "" },
        { id: 3, comp: <Team />, sectionID: "team", sectionCLass: "", dataAOS: "fade-up", sectionTitle: "Team", sectionDescription: "" },
        { id: 4, comp: <Contact />, sectionID: "contact", sectionCLass: "", sectionTitle: "Contact", sectionDescription: "" },

      ]
    }
  }

  DisplaySection(section) {
    if (section.sectionID === "hero") {
      return (
        <>
          <section id={section.sectionID}>
            <div className="hero-container" data-aos={section.dataAOS} data-aos-delay={section.dataAosDelay}>
              <h1>{section.sectionTitle}</h1>
              <h2>{section.sectionDescription}</h2>
              <a href="#chatbot" className="btn-get-started">{section.btnText}</a>
            </div>
          </section>
        </>
      );
    }
    else if (section.sectionID === "about") {
      return (
        <section id={section.sectionID} key={section.id}>
          <div className="container" data-aos={section.dataAOS}>
            <div className="row about-container">
              <div className="col-lg-6 content order-lg-1 order-2">
                <h2 className="title">{section.sectionTitle}</h2>
                <p>{section.sectionDescription}</p>
                {section.comp}
              </div>
              <div className="col-lg-6 background order-lg-2 order-1" data-aos="fade-left" data-aos-delay="100"></div>
            </div>
          </div>
        </section>
      );
    }
    else {
      return (
        <section id={section.sectionID} key={section.id}>
          <div className="container" data-aos={section.dataAOS}>
            <div className="section-header">
              <h3 className="section-title">{section.sectionTitle}</h3>
              <p className="section-description">{section.sectionDescription}</p>
            </div>
            {section.comp}
          </div>
        </section>
      );
    }
  }
  render() {
    return (
      <>
        {this.state.sections.map(section => (
          this.DisplaySection(section)
        ))}

      </>

    );
  }
}

export default Section;