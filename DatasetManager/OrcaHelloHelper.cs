// Copyright (c) PODS-AI contributors
// SPDX-License-Identifier: MIT

using AIForOrcas.Client.BL.Services;
using AIForOrcas.DTO;
using AIForOrcas.DTO.API;
using Amazon.Runtime;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Navigation;
using System.Xml.XPath;

namespace DatasetManager
{
    public class NoAuthTokenProvider : IAuthTokenProvider
    {
        public string GetToken() => string.Empty;
    }

    public class OrcaHelloHelper
    {
        public static async Task<List<Detection>?> FetchAllDetectionsAsync(
            DateTime startTime,
            DateTime endTime)
        {
            var services = new ServiceCollection();

            services.AddHttpClient("UnauthenticatedAPI", (sp, client) =>
            {
                string apiUrl = "https://aifororcasdetections.azurewebsites.net/";
                client.BaseAddress = new Uri(apiUrl);
                client.Timeout = TimeSpan.FromSeconds(30);
            });
            services.AddLogging();

            services.AddSingleton<IAuthTokenProvider, NoAuthTokenProvider>();
            services.AddScoped<IApiClientHelper>(sp =>
                new ApiClientHelper(
                    sp.GetRequiredService<IHttpClientFactory>(),
                    sp.GetRequiredService<ILogger<ApiClientHelper>>()));
            services.AddTransient<DetectionService>();

            using var provider = services.BuildServiceProvider();
            var detectionService = provider.GetRequiredService<DetectionService>();

            var filterOptions = new ReviewedFilterOptionsDTO
            {
                SortOrder = "asc",
                SortBy = "timestamp",
                Timeframe = "range",
                Location = "all",
                HydrophoneId = "all",
                DateFrom = startTime,
                DateTo = endTime
            };

            var allDetections = new List<Detection>();
            int page = 1;
            int totalPages;

            do
            {
                var paginationOptions = new PaginationOptionsDTO
                {
                    Page = page,
                    MinutesPerPage = 50,
                };

                PaginatedResponseDTO<List<Detection>> result;
                try
                {
                    result = await detectionService.GetDetectionsAsync(paginationOptions, filterOptions);
                }
                catch (Exception)
                {
                    MessageBox.Show(
                        "Could not retrieve detections from OrcaHello. Please check your connection and try again.",
                        "OrcaHello update failed",
                        MessageBoxButton.OK,
                        MessageBoxImage.Error);
                    return null;
                }

                if (result?.Response == null)
                {
                    // Failed.
                    return null;
                }
                allDetections.AddRange(result.Response);
                totalPages = result.TotalAmountPages;
                page++;
            } while (page <= totalPages);

            return allDetections;
        }

        public static async Task UpdateModeratedTagsAsync(IEnumerable<SampleRecord> samples)
        {
            if (!samples.Any())
            {
                return;
            }

            List<SampleRecord> list = samples.ToList();
            DateTime startTime = samples.Min(s => s.StartTimestampUtc);
            DateTime endTime = samples.Max(s => s.StartTimestampUtc) + TimeSpan.FromMinutes(1); // Add 1 minute to include the last sample in the range.
            List<Detection>? detections = await FetchAllDetectionsAsync(startTime, endTime);
            if (detections == null)
            {
                return;
            }
            TimeSpan audioOffset = TimeSpan.FromSeconds(2); // The audio offset in seconds to match the sample timestamp with the detection timestamp.

            // For each sample, find the corresponding detection and update the ModeratedTags.
            foreach (var sample in samples)
            {
                var matchingDetection = detections.FirstOrDefault(d =>
                    d.Timestamp == sample.StartTimestampUtc + audioOffset &&
                    d.Location.Id == sample.NodeName);
                if (matchingDetection != null)
                {
                    sample.Confidence = matchingDetection.Confidence;
                    sample.ModeratedTags = matchingDetection.Tags;
                }
            }
        }
    }
}
