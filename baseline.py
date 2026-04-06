from openenv_support_env.baseline import parse_args, run_baseline


if __name__ == "__main__":
    args = parse_args()
    run_baseline(task_level=args.task)
