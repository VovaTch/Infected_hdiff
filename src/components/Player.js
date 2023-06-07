import React, { useState, useRef, useEffect } from 'react'
import PlayerDetails from './PlayerDetails';
import PlayerControls from './PlayerControls';

function Player(props) {

  // Setting audio element and playing state
  const audioEl = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentModel, setCurrentModel] = useState({
    name: "",
    num: 0,
    progress: 1,
  });
  const [firstClick, setFirstClick] = useState(true)

  // Use effect hooks
  useEffect(() => {
    if (isPlaying) {
      audioEl.current.play();
    } else {
      audioEl.current.pause();
    }
  })
  useEffect(() => {
    if (!firstClick) {
      props.addBar(currentModel.name, "#0015aa");
    }
  }, [currentModel, firstClick])

  // Create the window of the player
  return (
    <div className="c-player">

      <audio src={props.waveform_sound} ref={audioEl}></audio>
      <h4>Playing the music sample</h4>
      <PlayerDetails waveform_img={props.waveform_img} />
      <PlayerControls
        isPlaying={isPlaying}
        setIsPlaying={setIsPlaying}
        setCurrentModel={setCurrentModel}
        createFirstClick={setFirstClick}
      />

    </div>
  )
}

export default Player;