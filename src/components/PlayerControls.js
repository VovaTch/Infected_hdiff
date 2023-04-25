import React from 'react'

import {ReactComponent as PlayComponent} from './resources/play.svg'
import {ReactComponent as PauseComponent} from './resources/pause.svg'
import {ReactComponent as StopComponent} from './resources/stop.svg'
import {ReactComponent as DownloadComponent} from './resources/download.svg'
import {ReactComponent as CreateComponent} from './resources/create.svg'


function PlayerControls(props) {

  function RunCreation() { //A temporary function until the connection with Python is implemented
    
    const current_model = {
      name: "Level 4 Diffusion",
      num: 1,
      progress: 55,
    }

    return (current_model)

  }

  return (
    <div className='c-player--controls'>

        <button className="create-btn" onClick={() => {
          props.createFirstClick(false); 
          props.setCurrentModel(RunCreation());
          }}><CreateComponent/></button>

        <button className="play-btn" onClick={() => props.setIsPlaying(!props.isPlaying)}>
          {!props.isPlaying ? <PlayComponent/> : <PauseComponent/>}
        </button>

        <button className="stop-btn"><StopComponent/></button>
        
        <button className="download-btn"><DownloadComponent/></button>

    </div>
  )
}

export default PlayerControls