import React from 'react';
import { styled } from '@mui/material/styles';

const Instructions1 = styled("div")({
  backgroundColor: `rgba(14, 13, 14, 1)`,
  display: `flex`,
  position: `relative`,
  isolation: `isolate`,
  flexDirection: `column`,
  justifyContent: `flex-start`,
  alignItems: `center`,
  width: `100vw`,
  height: `100vh`,
  padding: `0px`,
  boxSizing: `border-box`,
  overflow: `hidden`,
});

const BgOpening = styled("img")({
  width: `100%`,
  height: `100%`,
  objectFit: `cover`,
  position: `fixed`,
  top: `0`,
  left: `0`,
  zIndex: `-1`,
});

const Instructions2 = styled("div")({
  textAlign: `center`,
  whiteSpace: `pre-wrap`,
  fontSynthesis: `none`,
  color: `rgba(241, 241, 241, 1)`,
  fontStyle: `normal`,
  fontFamily: `Anuphan`,
  fontWeight: `700`,
  fontSize: `96px`,
  letterSpacing: `0px`,
  textDecoration: `none`,
  textTransform: `uppercase`,
  background: `linear-gradient(90.51deg, rgba(80, 79, 82, 0.5) 51%, rgba(209, 207, 213, 0.5) 65%)`,
  backgroundClip: `text`,
  WebkitBackgroundClip: `text`,
  WebkitTextFillColor: `transparent`,
  width: `1144px`,
  height: `auto`,
  marginTop: `40px`,
  zIndex: `2`,
});

const FrameContainer = styled("div")({
  display: `flex`,
  position: `relative`,
  width: `820px`,
  height: `520px`,
  justifyContent: `space-between`,
  alignItems: `center`,
  marginTop: `60px`,
  zIndex: `2`,
});

const Stats = styled("img")({
  height: `auto`,
  width: `482px`,
  position: `absolute`,
  right: `60px`,
  top: `654px`,
});

const Arrow = styled("img")({
  height: `auto`,
  width: `auto`,
  position: `absolute`,
});

function Instructions() {
  return (
    <Instructions1>
      <BgOpening src="/images/bg-opening.png" alt="Background Opening" />
      
      <Instructions2>
        INSTRUCTIONS
      </Instructions2>

      <FrameContainer>
        <Arrow src="/images/arrow-left.png" alt="Left Arrow" style={{ left: '0' }} />
        {/* Center frame content can go here */}
        <Arrow src="/images/arrow-right.png" alt="Right Arrow" style={{ right: '0' }} />
      </FrameContainer>
      <Stats src="/images/landing_stats.svg" alt="Stats" />
    </Instructions1>
  );
}

export default Instructions;
