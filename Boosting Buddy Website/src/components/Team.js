import React, { Component } from 'react'

class Team extends Component {
  constructor(props) {
    super(props);
    this.state = {
      team: [
        {
          id: 1, dataIosDelay: "100", imgURL: "assets/img/team-1.jpg", name: "Muhammad Rehan", role: "AI Engineer",
          links: [
            { id: 1, linkURL: "/https://github.com/RehanKhn ", classes: "bi bi-github" },
            { id: 2, linkURL: "/https://www.facebook.com/devRehanK/", classes: "bi bi-facebook" },
            { id: 3, linkURL: "/https://www.instagram.com/dev_rehank/", classes: "bi bi-instagram" },
            { id: 4, linkURL: "/https://www.linkedin.com/in/dev-rehan/", classes: "bi bi-linkedin" }
          ]
        },
        {
          id: 2, dataIosDelay: "200", imgURL: "assets/img/team-2.jpg", name: "Aqsa Shehzad", role: "AI Engineer",
          links: [
            { id: 1, linkURL: "/https://github.com/DevAqsaShahzad", classes: "bi bi-github" },
            { id: 2, linkURL: "/https://www.facebook.com/aqsa.shahzad.560", classes: "bi bi-facebook" },
            { id: 3, linkURL: "/https://www.instagram.com/aqsa_khan07/", classes: "bi bi-instagram" },
            { id: 4, linkURL: "/https://www.linkedin.com/in/aqsashahzad/", classes: "bi bi-linkedin" }
          ]
        },
        {
          id: 3, dataIosDelay: "300", imgURL: "assets/img/team-3.jpg", name: "Saqlain Tahir", role: "AI Engineer",
          links: [
            { id: 1, linkURL: "/https://github.com/Dev-Saqlain", classes: "bi bi-github" },
            { id: 2, linkURL: "/https://www.facebook.com/saqlain.tahir.058", classes: "bi bi-facebook" },
            { id: 3, linkURL: "/https://www.instagram.com/saqlain_tahir/", classes: "bi bi-instagram" },
            { id: 4, linkURL: "/https://www.linkedin.com/in/muhammad-saqlain-tahir-/", classes: "bi bi-linkedin" }
          ]
        },
        {
          id: 4, dataIosDelay: "400", imgURL: "assets/img/team-4.jpg", name: "Afifa Aslam", role: "AI Engineer",
          links: [
            { id: 1, linkURL: "/https://github.com/Afifa-Aslam", classes: "bi bi-github" },
            { id: 2, linkURL: "/https://www.facebook.com/afifa.siddiqui.9231", classes: "bi bi-facebook" },
            { id: 3, linkURL: "/https://www.instagram.com/afifaaslamsiddiqui/", classes: "bi bi-instagram" },
            { id: 4, linkURL: "/https://www.linkedin.com/in/afifa-aslam-siddiqui-516aba185/", classes: "bi bi-linkedin" }
          ]
        }
      ]
    }
  }

  DisplayLinks(link){
    return(
      <a href={link.linkURL} key={link.id} ><i className={link.classes}></i></a>
    );
  }

  TeamCard(teamMember) {
    return (
      <div className="col-lg-3 col-md-6" key={teamMember.id}>
        <div className="member" data-aos="fade-up" data-aos-delay={teamMember.dataIosDelay}>
          <div className="pic"><img src={teamMember.imgURL} alt="" /></div>
          <h4>{teamMember.name}</h4>
          <span>{teamMember.role}</span>
          <div className="social">
            {teamMember.links.map(link => (
              this.DisplayLinks(link)
            ))}
          </div>
        </div>
      </div>
    );
  }
  render() {
    return (
      <div className="row">
        {this.state.team.map(teamMember => (
          this.TeamCard(teamMember)
        ))}
      </div>
    );
  }
}

export default Team;