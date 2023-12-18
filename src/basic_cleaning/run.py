#!/usr/bin/env python
"""
Performs basic cleaning on the data and save the results in Weights & Biases
"""
import argparse
import logging
import os

import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def find_outliers_by_IQR(data: pd.Series):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1

    lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return lower_bound, upper_bound


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using
    # this particular version of the artifact.
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    # download data from wandb
    df = pd.read_csv(artifact_local_path)
    logger.info("step[preprocess]: data downloaded!")
    # Drop Price outliers
    min_price = args.min_price
    max_price = args.max_price
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()
    # Drop number_of_reviews outliers
    min_num_of_reviews, max_num_of_reviews = find_outliers_by_IQR(
        df.number_of_reviews)  # NOQA:E501
    idx = df['number_of_reviews'].between(min_num_of_reviews,
                                          max_num_of_reviews)
    df = df[idx].copy()
    # Drop minimum_nights outliers
    min_minimum_nights, max_minimum_nights = find_outliers_by_IQR(
        df.minimum_nights)  # NOQA:E501
    idx = df['minimum_nights'].between(min_minimum_nights, max_minimum_nights)
    df = df[idx].copy()
    # Drop calculated_host_listings_count
    min_calculated_host_listings_count = 0
    max_calculated_host_listings_count = 10
    idx = df['calculated_host_listings_count'].between(
        min_calculated_host_listings_count,  # NOQA:E501
        max_calculated_host_listings_count)  # NOQA:E501
    df = df[idx].copy()
    # Drop longitude
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)  # NOQA:E501
    df = df[idx].copy()
    # Drop NA
    df = df.dropna(axis=0)
    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])
    logger.info("step[preprocess]: data preprocessed!")
    # save and upload processed data to wandb
    df.to_csv("clean_sample.csv", index=False)
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)
    logger.info("step[preprocess]: data uploaded!")

    artifact.wait()
    os.remove("clean_sample.csv")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This steps cleans the data")

    parser.add_argument("--input_artifact",
                        type=str,
                        help="path or url of the data need to be cleaned",
                        required=True)

    parser.add_argument("--output_artifact",
                        type=str,
                        help="path or url to store the data after processed",
                        required=True)

    parser.add_argument("--output_type",
                        type=str,
                        help="output data type",
                        required=True)

    parser.add_argument("--output_description",
                        type=str,
                        help="output data description",
                        required=True)

    parser.add_argument("--min_price",
                        type=float,
                        help="min price in data that allowed",
                        required=True)

    parser.add_argument("--max_price",
                        type=float,
                        help="max price in data that allowed",
                        required=True)
    args = parser.parse_args()

    go(args)
