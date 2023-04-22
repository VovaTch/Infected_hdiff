import React, {useState, useRef, useEffect} from 'react'
import PlayerDetails from './PlayerDetails';
import PlayerControls from './PlayerControls';

function Player(props) {

  // Setting audio element and playing state
  const audioEl = useRef(null);
  const [isPlaying, setIsPlaying] = useState(true);

  // Audio play effect
  useEffect(() => {
    if (isPlaying) {
      audioEl.current.play();
    } else {
      audioEl.current.pause();
    }
  })

  // Create the window of the player
  return (
    <div className="c-player">

        <audio src={props.waveform_sound} ref={audioEl}></audio>
        <h4>Playing the music sample</h4>
        <PlayerDetails waveform_img={props.waveform_img} />
        <PlayerControls  
          isPlaying={isPlaying}
          setIsPlaying={setIsPlaying}
        />
        
    </div>
  )
}

export default Player;