# Import neccessary modules
import argparse

# Import custom file
import initialize_model

# Get all arguments needed to initialize a model
def get_model_init_args():
    parser = argparse.ArgumentParser(description="SimpleVisionML - Text Detection and Recognition made simple.")
    parser.add_argument(
        "--use_gpu",
        type=bool,
        default=True,
        help="Use gpu (default: True)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Device (default: None)",
    )
    parser.add_argument(
        "--model_storage_directory",
        type=str,
        default=None,
        help="Directory for model (.pth) file",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["text", "object-detection", "semantic-segmentation"],
        default="text",
        help="What task to run",
    )
    parser.add_argument(
        "--detection",
        type=bool,
        choices=[True, False],
        default=True,
        help="Use text detection",
    )
    parser.add_argument(
        "--detection_network",
        type=str,
        default='craft',
        help="Detection networks",
    )
    parser.add_argument(
        "--recognition",
        action="store_true",
        help="Use text recognition",
    )
    parser.add_argument(
        "--recognition_network",
        type=str,
        default='standard',
        help="Recognition networks",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        choices=[True, False],
        default=True,
        help="Print detail/warning",
    )
    parser.add_argument(
        "-f",
        "--file",
        required=True,
        type=str,
        help="Input file",
    )
    parser.add_argument(
        "--detail",
        type=int,
        choices=[0, 1],
        default=1,
        help="simple output (default: 1)",
    )
    parser.add_argument(
        "--rotation_info",
        type=list,
        default=None,
        help="Allow EasyOCR to rotate each text box and return the one with the best confident score. Eligible values are 90, 180 and 270. For example, try [90, 180 ,270] for all possible text orientations.",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_model_init_args()
    SimpleModel = initialize_model.SimpleModel(
                            use_gpu=args.use_gpu,\
                            device=args.device,\
                            model_storage_directory=args.model_storage_directory,\
                            task=args.task, \
                            detection=args.detection,\
                            detection_network=args.detection_network,\
                            recognition=args.recognition,\
                            recognition_network=args.recognition_network,\
                            verbose=args.verbose)
    
    # Process the given file/files
    SimpleModel.processFile(args.file)

if __name__ == "__main__":
    main()