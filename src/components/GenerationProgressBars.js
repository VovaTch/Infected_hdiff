import React from "react";

const ProgressBar = (props) => {
  const {bg_color, completed} = props;
  return (
    <div>
      <div>
        <span>{`${completed}%`}</span>
      </div>
    </div>
  )
}

export default ProgressBar