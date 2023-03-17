import React from 'react'
import {ReactComponent as PlayComponent} from './resources/play.svg'
import {ReactComponent as PauseComponent} from './resources/pause.svg'
import {ReactComponent as StopComponent} from './resources/stop.svg'
import {ReactComponent as DownloadComponent} from './resources/download.svg'
import {ReactComponent as CreateComponent} from './resources/create.svg'

function PlayerControls() {
  return (
    <div className='c-player--controls'>
        <button className="create-btn"><CreateComponent/></button>
        <button className="play-btn"><PlayComponent/></button>
        <button className="pause-btn"><PauseComponent/></button>
        <button className="stop-btn"><StopComponent/></button>
        <button className="download-btn"><DownloadComponent/></button>
    </div>
  )
}

export default PlayerControls