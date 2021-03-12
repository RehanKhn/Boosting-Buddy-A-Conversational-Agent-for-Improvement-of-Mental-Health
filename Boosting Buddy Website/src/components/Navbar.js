import React, { Component } from 'react';

class Navbar extends Component {
  constructor(props) {
    super(props);
    this.state = {
      navItems: [
        {
          id: 1,
          name: 'Home',
          link: '/#hero',
          class: 'nav-link scrollto'
        },
        {
          id: 2,
          name: 'Chatbot',
          link: '/#chatbot',
          class: 'nav-link scrollto'
        },
        {
          id: 3,
          name: 'About',
          link: '/#about',
          class: 'nav-link scrollto'
        },
        {
          id: 4,
          name: 'Team',
          link: '/#team',
          class: 'nav-link scrollto'
        },
        {
          id: 5,
          name: 'Contact',
          link: '/#contact',
          class: 'nav-link scrollto'
        }

      ]
    }
  }

  DisplayNavLinks(navItem) {

    return (
      <li key={navItem.id}><a className={navItem.class} href={navItem.link}>{navItem.name}</a></li>
    );

  }

  render() {
    return (
      <nav id="navbar" className="navbar">
        <ul>
          {this.state.navItems.map(navItem => (
            this.DisplayNavLinks(navItem)
          ))}

        </ul>

        <i className="bi bi-list mobile-nav-toggle"></i>
      </nav>
    );
  }
}

export default Navbar;
