import React, {useState} from "react";

const ProgressBar = (props) => {

  // Set bar settings and completed objects
  const [barSettings, setBarSettings] = useState(
    props.barSettings
  );
  const [barCompleted, setBarCompleted] = useState(props.barCompleted);
  const [id, setId] = useState(props.id);

  // Set custom values for style
  const fillerStyle = {
    backgroundColor: barSettings.color,
    width: `${barCompleted}%`,
  } 

  return (
    <div className="progress-bar--box">
      <h4>{barSettings.name}</h4>
      <div className="progress-bar--container" id={barSettings.name}>
        <div className="progress-bar--filler" style={fillerStyle}>
          <span className="progress-bar--label">{`${barCompleted}%`}</span>
        </div>
      </div>
    </div>
  )

}


export default ProgressBar