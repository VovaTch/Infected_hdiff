import React from 'react'
import { ProgressBar } from 'react-bootstrap'

function GenerationProgressBar(props) {
  return (
    <div>
        <ProgressBar variant={props.name} now={props.value}/>
    </div>
  )
}

export default GenerationProgressBars