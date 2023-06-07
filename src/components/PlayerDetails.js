import React from 'react'

function PlayerDetails(props) {
  return (
    <div className='c-player--details'>
      <div className="detail_img">
        <img src={props.waveform_img} alt="" />
      </div>
      <h3 className="details-level"></h3>
    </div>
  )
}

export default PlayerDetails