import React from 'react';
import { Link } from 'react-router-dom';  // Import Link for navigation
import { styled } from '@mui/material/styles';

const Landing1 = styled("div")({
  backgroundColor: `rgba(14, 13, 14, 1)`,
  display: `flex`,
  position: `relative`,
  isolation: `isolate`,
  flexDirection: `row`,
  width: `100vw`,  // Make the main container cover the entire viewport width
  height: `100vh`,  // Make the main container cover the entire viewport height
  justifyContent: `flex-start`,
  alignItems: `flex-start`,
  padding: `0px`,
  boxSizing: `border-box`,
  overflow: `hidden`,
});

const BgOpening = styled("img")({
  width: `100%`,  // Make the background image fill the width
  height: `100%`,  // Make the background image fill the height
  objectFit: `cover`,  // Ensure the image covers the screen while maintaining aspect ratio
  position: `fixed`,  // Keep the background fixed in place while scrolling/resizing
  top: `0`,
  left: `0`,
  zIndex: `-1`,  // Ensure it stays behind all other elements
});

const Jupiter = styled("img")({
  height: `auto`,
  width: `768px`,
  objectFit: `cover`,
  position: `absolute`,
  left: `682px`,
  top: `80px`,
});

const PWehi = styled("div")({
  textAlign: `right`,
  whiteSpace: `pre-wrap`,
  fontSynthesis: `none`,
  color: `rgba(0, 0, 0, 1)`,
  fontStyle: `normal`,
  fontFamily: `Anuphan`,
  fontWeight: `700`,
  fontSize: `15vw`,  // Adjust to viewport width for responsiveness
  letterSpacing: `0px`,
  textDecoration: `none`,
  textTransform: `uppercase`,
  background: `linear-gradient(0.8100000000000023deg, rgba(18, 17, 18, 1) -140.89664342631104%, rgba(135, 133, 138, 1) 145.05196262383006%)`,
  backgroundClip: `text`,
  WebkitBackgroundClip: `text`,
  WebkitTextFillColor: `transparent`,
  width: `auto`,  // Auto width based on content
  position: `absolute`,
  right: `70px`,
  top: `10vh`,  // Adjusted to make space for the full text
});

const Telescope = styled("img")({
  height: `1001px`,
  width: `1335px`,
  objectFit: `cover`,
  position: `absolute`,
  left: `auto`,
  top: `60px`,
});

const EnterACosmicSymphony = styled("div")({
  textAlign: `right`,
  whiteSpace: `pre-wrap`,
  fontSynthesis: `none`,
  color: `rgba(241, 241, 241, 1)`,
  fontStyle: `normal`,
  fontFamily: `Stick No Bills`,
  fontWeight: `400`,
  fontSize: `40px`,
  letterSpacing: `-0.25px`,
  textDecoration: `none`,
  lineHeight: `25px`,
  textTransform: `none`,
  width: `471px`,
  height: `43px`,
  position: `absolute`,
  left: `724px`,
  top: `297px`,
});

const Stats = styled("img")({
  height: `auto`,
  width: `482px`,
  position: `absolute`,
  right: `60px`,
  top: `654px`,
});

const Settings = styled("img")({
  height: `auto`,
  width: `26px`,
  position: `absolute`,
  left: `39px`,
  top: `42px`,
});

const ButtonPlay = styled("img")({
  width: `220px`,
  height: `65px`,
  position: `absolute`,
  left: `900px`,
  top: `340px`,
  cursor: `pointer`,  // Make it look clickable
});

const PlayAsGuest = styled("img")({
  width: `auto`,
  height: `auto`,
  position: `absolute`,
  right: `400px`,
  top: `415px`,
  cursor: `pointer`,  // Make it look clickable
});

function Landing() {
  return (
    <Landing1>
      <BgOpening src="/images/bg-opening.png" alt="Background Opening" />
      <Settings src="/images/landing_settings.svg" alt="Settings" />
      <Jupiter src="/images/landing_jupiter.svg" alt="Jupiter" />
      <PWehi>PŌWEHI</PWehi>
      <Telescope src="/images/landing_telescope.png" alt="Telescope" />
      <EnterACosmicSymphony>Enter a Cosmic Symphony</EnterACosmicSymphony>
      

      {/* Wrap the Play button with a Link */}
      <Link to="/instructions">
        <ButtonPlay src="/images/button-play.png" alt="Play Button" />
      </Link>
      <Stats src="/images/landing_stats.svg" alt="Stats" />
    </Landing1>
  );
}

export default Landing;
