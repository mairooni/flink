/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.table.examples.java.connectors;

import org.apache.flink.table.api.EnvironmentSettings;
import org.apache.flink.table.api.TableEnvironment;

/**
 * Example that calls an OpenAI chat completions model from Flink SQL to classify the sentiment of
 * movie reviews.
 *
 * <p>In particular, the example shows how to
 *
 * <ul>
 *   <li>declare a remote LLM as a catalog object with {@code CREATE MODEL},
 *   <li>describe its input and output schema so the planner can type-check it,
 *   <li>and invoke it row-by-row from a query with the {@code ML_PREDICT} table function.
 * </ul>
 *
 * <p>The model is not part of the job graph. {@code ML_PREDICT} issues one asynchronous HTTP
 * request per row against the configured endpoint, so the job needs network access and a valid API
 * key.
 *
 * <p><b>Running it.</b> The {@code flink-model-openai} provider is loaded through Java's service
 * loader at runtime and is deliberately not a compile-time dependency of this module. Add {@code
 * flink-models/flink-model-openai/target/flink-model-openai-{@literal <version>}.jar} to the
 * classpath of the run configuration, then set at least:
 *
 * <pre>{@code
 * OPENAI_API_KEY=sk-...          # required
 * OPENAI_ENDPOINT=...            # optional, defaults to the public OpenAI endpoint
 * OPENAI_MODEL=...               # optional, defaults to gpt-4o-mini
 * }</pre>
 *
 * <p>Pointing {@code OPENAI_ENDPOINT} at any OpenAI-compatible {@code /chat/completions} URL works
 * too, which is a convenient way to try the example without spending tokens.
 *
 * <p>The results are written to stdout.
 */
public final class OpenAIModelExample {

    private static final String DEFAULT_ENDPOINT = "https://api.openai.com/v1/chat/completions";

    private static final String DEFAULT_MODEL = "gpt-4o-mini";

    // *************************************************************************
    //     PROGRAM
    // *************************************************************************

    public static void main(String[] args) throws Exception {

        final String apiKey = System.getenv("OPENAI_API_KEY");
        if (apiKey == null || apiKey.isEmpty()) {
            throw new IllegalStateException(
                    "Set the OPENAI_API_KEY environment variable to run this example.");
        }
        final String endpoint = getEnvOrDefault("OPENAI_ENDPOINT", DEFAULT_ENDPOINT);
        final String model = getEnvOrDefault("OPENAI_MODEL", DEFAULT_MODEL);

        // set up the Table API; the source below is bounded, so the job terminates on its own
        final EnvironmentSettings settings =
                EnvironmentSettings.newInstance().inStreamingMode().build();
        final TableEnvironment tableEnv = TableEnvironment.create(settings);

        // declare the remote model
        //
        // the input schema must be a single STRING column, and for chat completions the output
        // schema must be a single STRING column as well; the system prompt is what turns a
        // general-purpose chat model into a sentiment classifier
        tableEnv.executeSql(
                String.format(
                        "CREATE MODEL ai_analyze_sentiment%n"
                                + "INPUT (`input` STRING)%n"
                                + "OUTPUT (`content` STRING)%n"
                                + "WITH (%n"
                                + "  'provider' = 'openai',%n"
                                + "  'endpoint' = '%s',%n"
                                + "  'api-key' = '%s',%n"
                                + "  'model' = '%s',%n"
                                + "  'system-prompt' = 'Classify the text below into one of the "
                                + "following labels: [positive, negative, neutral, mixed]. "
                                + "Output only the label.'%n"
                                + ")",
                        endpoint, apiKey, model));

        // a small bounded table of reviews to classify, with the label we expect back
        tableEnv.executeSql(
                "CREATE TEMPORARY VIEW movie_comment (id, movie_name, user_comment, actual_label)\n"
                        + "AS VALUES\n"
                        + "  (1, 'Am I Ok?', 'The most romantic storytelling I have seen in "
                        + "a long while. Gentle and full of love.', 'positive'),\n"
                        + "  (2, 'The Grey Hour', 'Two hours I will never get back. The plot "
                        + "goes nowhere at all.', 'negative'),\n"
                        + "  (3, 'Harbour Lights', 'It is a film. It has actors in it. Things "
                        + "happen.', 'neutral'),\n"
                        + "  (4, 'Second Spring', 'Gorgeous cinematography wasted on a script "
                        + "that cannot decide what it wants.', 'mixed')");

        // ML_PREDICT takes the input table, the model, and the column to feed into it; it returns
        // the input row widened by the model's output columns -- here the `content` column
        tableEnv.executeSql(
                        "SELECT id, movie_name, content AS predicted_label, actual_label\n"
                                + "FROM ML_PREDICT(\n"
                                + "  TABLE movie_comment,\n"
                                + "  MODEL ai_analyze_sentiment,\n"
                                + "  DESCRIPTOR(user_comment))")
                .print();
    }

    private static String getEnvOrDefault(String name, String defaultValue) {
        final String value = System.getenv(name);
        return value == null || value.isEmpty() ? defaultValue : value;
    }
}
