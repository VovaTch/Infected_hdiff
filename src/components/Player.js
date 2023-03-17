import React from 'react'
import PlayerDetails from './PlayerDetails';
import PlayerControls from './PlayerControls';

function Player(props) {

  return (
    <div className="c-player">

        <audio></audio>
        <h4>Playing the music sample</h4>
        <PlayerDetails waveform_img={props.waveform_img} />
        <PlayerControls/>
        
    </div>
  )
}

export default Player;