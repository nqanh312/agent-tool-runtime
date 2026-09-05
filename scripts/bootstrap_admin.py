"""Create or activate the first administrator without a default password."""

import argparse
import getpass

from services.auth import auth_repository, validate_password


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap the RBAC administrator")
    parser.add_argument("--username", default="admin")
    parser.add_argument("--display-name", default="Administrator")
    args = parser.parse_args()

    password = getpass.getpass("New administrator password: ")
    confirmation = getpass.getpass("Confirm password: ")
    if password != confirmation:
        raise SystemExit("Passwords do not match")
    validate_password(password)
    user = auth_repository.bootstrap_admin(args.username, args.display_name, password)
    print(f"Administrator '{user['username']}' is ready (id={user['user_id']}).")


if __name__ == "__main__":
    main()
