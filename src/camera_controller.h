#ifndef CAMERA_CONTROLLER_H
#define CAMERA_CONTROLLER_H

#include "camera_state.h"

class CameraController
{
public:
	CameraController() = default;

	void Update(
		const CameraInput& input,
		float deltaTime
	);

	const CameraState& GetState() const { return state; };
private:
	CameraState state;
};

#endif // !CAMERA_CONTROLLER_H
