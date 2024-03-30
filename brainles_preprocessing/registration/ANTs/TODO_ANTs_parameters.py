import os
import shlex

from app.project_e.image_processing.utilities.utils import eleSubprocess
from flask_socketio import SocketIO


def ants_registrator(
    fixed_image, moving_image, outputmat, transformationalgorithm="rigid"
):
    # ants call parameters
    dimensionality = "-d 3"
    initial_moving_transform = "-r [" + fixed_image + ", " + moving_image + ", 0]"

    # transformations
    if transformationalgorithm == "rigid":
        # rigid ants_transformation
        transform_rigid = "-t rigid[0.1]"
        metric_rigid = (
            "-m Mattes[" + fixed_image + "," + moving_image + ", 1, 32, Regular, 0.5]"
        )
        convergence_rigid = "-c [1000x500x250, 1e-6, 10]"
        smoothing_sigmas_rigid = "-s 3x2x1vox"
        shrink_factors_rigid = "-f 8x4x2"
    elif transformationalgorithm == "rigid+affine":
        # rigid ants_transformation
        transform_rigid = "-t rigid[0.1]"
        metric_rigid = (
            "-m Mattes[" + fixed_image + "," + moving_image + ", 1, 32, Regular, 0.5]"
        )
        convergence_rigid = "-c [1000x500x250, 1e-6, 10]"
        smoothing_sigmas_rigid = "-s 3x2x1vox"
        shrink_factors_rigid = "-f 8x4x2"

        # affine ants_transformation
        transform_affine = "-t affine[0.1]"
        metric_affine = (
            "-m Mattes[" + fixed_image + "," + moving_image + ", 1, 32, Regular, 0.5]"
        )
        convergence_affine = "-c [1000x500x250, 1e-6, 10]"
        smoothing_sigmas_affine = "-s 3x2x1vox"
        shrink_factors_affine = "-f 8x4x2"
    elif transformationalgorithm == "rex-dfc":
        # translation
        transform_translation = "-t translation[0.1]"
        metric_translation = (
            "-m Mattes[" + fixed_image + "," + moving_image + ", 1, 32, Regular, 0.05]"
        )
        convergence_translation = "-c [1000, 1e-8, 20]"
        smoothing_sigmas_translation = "-s 4vox"
        shrink_factors_translation = "-f 6"

        # rigid ants_transformation
        transform_rigid = "-t rigid[0.1]"
        metric_rigid = (
            "-m Mattes[" + fixed_image + "," + moving_image + ", 1, 32, Regular, 0.1]"
        )
        convergence_rigid = "-c [1000x1000, 1e-8, 20]"
        smoothing_sigmas_rigid = "-s 4x2vox"
        shrink_factors_rigid = "-f 4x2"

        # affine ants_transformation
        transform_affine = "-t affine[0.1]"
        metric_affine = (
            "-m Mattes[" + fixed_image + "," + moving_image + ", 1, 32, Regular, 0.1]"
        )
        convergence_affine = "-c [10000x1111x5, 1e-8, 20]"
        smoothing_sigmas_affine = "-s 3x2x1vox"
        shrink_factors_affine = "-f 8x4x2"

    # other parameters
    use_estimate_learning_rate_once = "-l 1"
    collapse_output_transforms = "-z 1"
    interpolation = "-n BSpline[3]"
    precision = "--float 1"
    output = "-o " + "[" + outputmat + "]"

    # generate calls
    if transformationalgorithm == "rigid":
        ants_cmd = (
            "antsRegistration",
            dimensionality,
            initial_moving_transform,
            # rigid ants_transformation
            transform_rigid,
            metric_rigid,
            convergence_rigid,
            smoothing_sigmas_rigid,
            shrink_factors_rigid,
            # other parameters
            use_estimate_learning_rate_once,
            collapse_output_transforms,
            interpolation,
            precision,
            output,
        )
        ants_call = shlex.split("%s %s %s %s %s %s %s %s %s %s %s %s %s" % ants_cmd)

    elif transformationalgorithm == "rigid+affine":
        ants_cmd = (
            "antsRegistration",
            dimensionality,
            initial_moving_transform,
            # rigid ants_transformation
            transform_rigid,
            metric_rigid,
            convergence_rigid,
            smoothing_sigmas_rigid,
            shrink_factors_rigid,
            # affine ants_transformation
            transform_affine,
            metric_affine,
            convergence_affine,
            smoothing_sigmas_affine,
            shrink_factors_affine,
            # other parameters
            use_estimate_learning_rate_once,
            collapse_output_transforms,
            interpolation,
            precision,
            output,
        )
        ants_call = shlex.split(
            "%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s" % ants_cmd
        )

    elif transformationalgorithm == "rex-dfc":
        ants_cmd = (
            "antsRegistration",
            dimensionality,
            initial_moving_transform,
            # translation
            transform_translation,
            metric_translation,
            convergence_translation,
            smoothing_sigmas_translation,
            shrink_factors_translation,
            # rigid ants_transformation
            transform_rigid,
            metric_rigid,
            convergence_rigid,
            smoothing_sigmas_rigid,
            shrink_factors_rigid,
            # affine ants_transformation
            transform_affine,
            metric_affine,
            convergence_affine,
            smoothing_sigmas_affine,
            shrink_factors_affine,
            # other parameters
            use_estimate_learning_rate_once,
            collapse_output_transforms,
            interpolation,
            precision,
            output,
        )
        ants_call = shlex.split(
            "%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s"
            % ants_cmd
        )

    # construct call
    readable_ants_call = " ".join(ants_cmd)
    print("calling ants with the following call:")
    print(readable_ants_call)

    # log file
    logFilePath = outputmat + "registration.log"
    # call it
    eleSubprocess(logFilePath=logFilePath, call=ants_call)


def modality_registrator(examid, modality):
    socketio = SocketIO(message_queue="redis://")
    socketio.emit(
        "ipstatus", {"examid": examid, "ipstatus": modality + " ants registration"}
    )

    niftipath = os.path.normpath(os.path.join("data/tmp/", examid, "raw_niftis"))
    # fixed image
    native_t1 = os.path.join(niftipath, examid + "_native_t1.nii.gz")

    # moving images
    moving_image = os.path.join(niftipath, examid + "_native_" + modality + ".nii.gz")

    # output mats
    exportpath = os.path.normpath(os.path.join("data/tmp/", examid, "registrations"))
    os.makedirs(exportpath, exist_ok=True)
    filename = examid + "_" + modality + "_to_t1_"
    outputmat = os.path.join(exportpath, filename)

    # call it
    ants_registrator(
        native_t1, moving_image, outputmat, transformationalgorithm="rigid"
    )
