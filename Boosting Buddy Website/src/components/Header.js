import Logo from './Logo'
import Navbar from "./Navbar";


function Header() {
  return (
    <header id="header" className="fixed-top d-flex align-items-center ">
      <div className="container d-flex justify-content-between align-items-center">
        <Logo />
        <Navbar />
      </div>
    </header>
  );
}

export default Header;