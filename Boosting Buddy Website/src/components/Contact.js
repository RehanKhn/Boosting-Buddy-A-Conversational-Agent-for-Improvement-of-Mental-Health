import React, { Component } from 'react';


class Contact extends Component {
  constructor(props) {
    super(props);
    this.state = {
      info: [
        { id: 0, class: "bi bi-geo-alt", dept: "Department of Computer Science", uni: "University of Engineering and Technology Lahore." },
        { id: 1, class: "bi bi-envelope", email: "cs@uet.edu.pk" },
        { id: 2, class: "bi bi-phone", phone: "+4237951901" }
      ],

      inputs: [
        { id: 1, divClass: "form-group", type: "text", name: "name", inputId: "name", placeholder: "Your Name" },
        { id: 2, divClass: "form-group mt-3", type: "email", name: "email", inputId: "email", placeholder: "Your Email" },
        { id: 3, divClass: "form-group mt-3", type: "text", name: "subject", inputId: "subject", placeholder: "Subject" },
        { id: 4, divClass: "form-group mt-3", name: "message", row: "5", placeholder: "Message" },
      ]
    }
  }
  DisplayInfo(info) {
    if (info.id === 0) {
      return (
        <div key={info.id}>
          <i className={info.class}></i>
          <p>{info.dept}<br />{info.uni}</p>
        </div>
      );
    }
    else if (info.id === 1) {
      return (
        <div key={info.id}>
          <i className={info.class}></i>
          <p>{info.email}</p>
        </div>
      );
    }
    else {
      return (
        <div key={info.id}>
          <i className={info.class}></i>
          <p>{info.phone}</p>
        </div>
      );
    }
  }



  DisplayForm(input) {
    if (input.id === 4) {
      return (
        <>
          <div className={input.divClass}>
            <textarea className="form-control" name={input.name} rows={input.row} placeholder={input.placeholder} required></textarea>
          </div>
        </>
      );
    }
    else {
      return (
        <>
          <div className={input.divClass}>
            <input type={input.type} name={input.name} className="form-control" id={input.inputId} placeholder={input.placeholder} required />
          </div>
        </>
      );
    }
  }
  render() {
    return (

      <div className="container mt-5">
        <div className="row justify-content-center">

          <div className="col-lg-3 col-md-4">

            <div className="info">
              {this.state.info.map(info => (
                this.DisplayInfo(info)
              ))}
            </div>


          </div>

          <div className="col-lg-5 col-md-8">
            <div className="form">
              <form action="forms/contact.php" method="post" role="form" className="php-email-form">
                {this.state.inputs.map(input => (
                  this.DisplayForm(input)
                ))}
                <div className="my-3">
                  <div className="loading">Loading</div>
                  <div className="error-message"></div>
                  <div className="sent-message">Your message has been sent. Thank you!</div>
                </div>
                <div className="text-center"><button type="submit">Send Message</button></div>

              </form>
            </div>
          </div>

        </div>

      </div>

    );
  }
}

export default Contact;