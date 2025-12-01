
import argparse
import os

from simulate import simulate_customers
from signal_model import compute_customer_signals


def main():
	parser = argparse.ArgumentParser(description='Early Risk Signal prototype')
	parser.add_argument('--simulate', action='store_true', help='Run data simulation')
	parser.add_argument('--n', type=int, default=1000, help='Number of customers to simulate')
	parser.add_argument('--months', type=int, default=6, help='Number of months to simulate')
	parser.add_argument('--out', type=str, default='scores.csv', help='Output CSV file for scores')
	args = parser.parse_args()

	if args.simulate:
		print(f"Simulating {args.n} customers for {args.months} months...")
		df = simulate_customers(n_customers=args.n, months=args.months)
		print("Computing signals...")
		summary = compute_customer_signals(df)
		out_path = os.path.abspath(args.out)
		summary.to_csv(out_path, index=False)
		print(f"Wrote scores to {out_path}")
	else:
		parser.print_help()


if __name__ == '__main__':
	main()
